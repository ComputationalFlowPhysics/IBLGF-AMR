#!/usr/bin/env python3
"""
fix_config.py

Usage:
  python3 fix_config.py INPUT_CONFIG [OUTPUT_CONFIG]

Fixes three constraints in an IBLGF-style config file:
  (1) For each dimension: domain.bd_extent[d] / domain.block_extent is an integer power of 2
      - If violated, adjusts domain.bd_extent to the closest valid multiple of block_extent.
  (2) domain.block.extent == domain.bd_extent
  (3) domain.block.base   == domain.bd_base

Prints a report of changes.
"""

from __future__ import annotations

import os
import re
import sys
from collections import OrderedDict
from typing import Any, List, Tuple, Union, Optional

Number = Union[int, float]
Value = Union[Number, bool, str, Tuple[Any, ...]]


# ----------------------------
# Helpers
# ----------------------------

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def nearest_power_of_two_int(x: float) -> int:
    """Nearest power-of-two integer to x (ties -> higher)."""
    if x <= 1:
        return 1
    lo = 1
    while lo * 2 <= x:
        lo *= 2
    hi = lo * 2
    return lo if abs(x - lo) < abs(hi - x) else hi


def parse_scalar(s: str) -> Value:
    s = s.strip()

    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if not inner:
            return tuple()
        parts = [p.strip() for p in inner.split(",")]
        return tuple(parse_scalar(p) for p in parts)

    if s == "true":
        return True
    if s == "false":
        return False

    # quoted
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    # int
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)

    # float / sci
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", s):
        return float(s)

    return s


def format_value(v: Value) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, tuple):
        return "(" + ",".join(format_value(x) for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.15g}"
    if isinstance(v, int):
        return str(v)
    return str(v)


class Block:
    def __init__(self) -> None:
        self.items: "OrderedDict[str, Any]" = OrderedDict()

    def get_path(self, path: str) -> Any:
        parts = path.split(".")
        cur: Any = self
        for p in parts:
            if not isinstance(cur, Block):
                raise KeyError(f"Path '{path}' goes through a non-block at '{p}'.")
            if p not in cur.items:
                raise KeyError(f"Key '{p}' not found in path '{path}'.")
            cur = cur.items[p]
        return cur

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        cur: Any = self
        for p in parts[:-1]:
            if p not in cur.items or not isinstance(cur.items[p], Block):
                cur.items[p] = Block()
            cur = cur.items[p]
        cur.items[parts[-1]] = value

    def to_text(self, indent: int = 0, name: Optional[str] = None) -> str:
        sp = " " * indent
        lines: List[str] = []
        if name is not None:
            lines.append(f"{sp}{name}")
            lines.append(f"{sp}" + "{")
            body_indent = indent + 4
        else:
            body_indent = indent

        bsp = " " * body_indent
        for k, v in self.items.items():
            if isinstance(v, Block):
                lines.append(v.to_text(body_indent, k).rstrip("\n"))
            else:
                lines.append(f"{bsp}{k}={format_value(v)};")

        if name is not None:
            lines.append(f"{sp}" + "}")
        return "\n".join(lines) + "\n"


def tokenize_config(text: str) -> List[str]:
    # strip // comments
    text = re.sub(r"//.*", "", text)

    tokens: List[str] = []
    i = 0
    n = len(text)

    def skip_ws(j: int) -> int:
        while j < n and text[j].isspace():
            j += 1
        return j

    while i < n:
        i = skip_ws(i)
        if i >= n:
            break

        ch = text[i]
        if ch in "{}=;":
            tokens.append(ch)
            i += 1
            continue

        # tuple "( ... )" as one token (supports nested parentheses but not needed here)
        if ch == "(":
            depth = 0
            j = i
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            tokens.append(text[i:j].strip())
            i = j
            continue

        # bare token up to whitespace or delimiter
        j = i
        while j < n and (not text[j].isspace()) and text[j] not in "{}=;":
            j += 1
        tok = text[i:j].strip()
        if tok:
            tokens.append(tok)
        i = j

    return tokens


def parse_config(text: str) -> Block:
    toks = tokenize_config(text)
    pos = 0

    def expect(tok: str) -> None:
        nonlocal pos
        got = toks[pos] if pos < len(toks) else "<EOF>"
        if got != tok:
            raise ValueError(f"Expected '{tok}' but got '{got}' at token {pos}.")
        pos += 1

    def parse_block_contents() -> Block:
        nonlocal pos
        b = Block()
        while pos < len(toks) and toks[pos] != "}":
            key = toks[pos]
            pos += 1
            if pos < len(toks) and toks[pos] == "{":
                pos += 1
                child = parse_block_contents()
                expect("}")
                b.items[key] = child
            else:
                expect("=")
                val_tok = toks[pos]
                pos += 1
                expect(";")
                b.items[key] = parse_scalar(val_tok)
        return b

    root = Block()
    while pos < len(toks):
        name = toks[pos]
        pos += 1
        expect("{")
        blk = parse_block_contents()
        expect("}")
        root.items[name] = blk

    return root


def as_int_tuple(v: Value) -> Tuple[int, ...]:
    if isinstance(v, tuple):
        return tuple(int(x) for x in v)
    if isinstance(v, (int, float)):
        x = int(v)
        return (x, x, x)
    raise ValueError(f"Expected tuple or number, got {v!r}")


def closest_valid_extent(cur: int, block_extent: int) -> int:
    """
    Choose closest extent = block_extent * (power-of-two int blocks).
    """
    if block_extent <= 0:
        return cur

    ratio = cur / block_extent
    # candidate powers: nearest and neighbors
    n0 = nearest_power_of_two_int(ratio)
    cand = []
    for nb in {max(1, n0 // 2), n0, n0 * 2}:
        cand.append(block_extent * nb)

    best = min(cand, key=lambda x: abs(x - cur))
    return best


# ----------------------------
# Main fixing logic
# ----------------------------

def fix_domain_constraints(sim: Block) -> List[str]:
    """
    Applies 3 constraints:
      (1) bd_extent/block_extent is integer power-of-two per dimension (adjust bd_extent if needed)
      (2) domain.block.extent == domain.bd_extent
      (3) domain.block.base   == domain.bd_base
    Returns human-readable change messages.
    """
    changes: List[str] = []

    # Required keys
    bd_base_path = "domain.bd_base"
    bd_extent_path = "domain.bd_extent"
    block_extent_path = "domain.block_extent"

    bd_base = sim.get_path(bd_base_path)
    bd_extent = sim.get_path(bd_extent_path)
    block_extent = int(sim.get_path(block_extent_path))

    bd_base_t = as_int_tuple(bd_base)
    bd_extent_t = as_int_tuple(bd_extent)

    # Enforce same dimensionality between base and extent (use extent dim as authority)
    dim = len(bd_extent_t)
    if len(bd_base_t) != dim:
        # pad or truncate bd_base to match bd_extent dimension
        old = bd_base_t
        if len(old) < dim:
            bd_base_t = old + (0,) * (dim - len(old))
        else:
            bd_base_t = old[:dim]
        sim.set_path(bd_base_path, bd_base_t)
        changes.append(f"Adjusted domain.bd_base dimensionality {old} -> {bd_base_t} to match domain.bd_extent dim={dim}.")

    # (1) power-of-two blocks: bd_extent[d]/block_extent = 2^k integer
    new_extent = list(bd_extent_t)
    for d in range(dim):
        cur = bd_extent_t[d]
        if block_extent <= 0:
            break
        ratio = cur / block_extent
        ok = float(ratio).is_integer() and is_power_of_two(int(ratio))
        if not ok:
            fixed = closest_valid_extent(cur, block_extent)
            new_ratio = fixed / block_extent
            changes.append(
                f"domain.bd_extent[{d}] {cur} -> {fixed} "
                f"(blocks: {ratio:g} -> {new_ratio:g}, block_extent={block_extent})"
            )
            new_extent[d] = fixed

    new_extent_t = tuple(new_extent)
    if new_extent_t != bd_extent_t:
        sim.set_path(bd_extent_path, new_extent_t)
        bd_extent_t = new_extent_t  # update in-memory

    # (2) sync domain.block.extent
    try:
        old_block_extent = sim.get_path("domain.block.extent")
        old_block_extent_t = as_int_tuple(old_block_extent)
        if old_block_extent_t != bd_extent_t:
            sim.set_path("domain.block.extent", bd_extent_t)
            changes.append(f"domain.block.extent {old_block_extent_t} -> {bd_extent_t} (synced to domain.bd_extent)")
    except KeyError:
        # If missing, create it (harmless and often helpful)
        sim.set_path("domain.block.extent", bd_extent_t)
        changes.append(f"domain.block.extent was missing; set to {bd_extent_t} (synced to domain.bd_extent)")

    # (3) sync domain.block.base
    try:
        old_block_base = sim.get_path("domain.block.base")
        old_block_base_t = as_int_tuple(old_block_base)
        if old_block_base_t != bd_base_t:
            sim.set_path("domain.block.base", bd_base_t)
            changes.append(f"domain.block.base {old_block_base_t} -> {bd_base_t} (synced to domain.bd_base)")
    except KeyError:
        sim.set_path("domain.block.base", bd_base_t)
        changes.append(f"domain.block.base was missing; set to {bd_base_t} (synced to domain.bd_base)")

    return changes


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 2

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.isfile(in_path):
        print(f"ERROR: Input file not found: {in_path}")
        return 2

    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()

    root = parse_config(text)
    if "simulation_parameters" not in root.items:
        print("ERROR: No 'simulation_parameters { ... }' block found.")
        return 2

    sim = root.items["simulation_parameters"]

    try:
        changes = fix_domain_constraints(sim)
    except KeyError as e:
        print(f"ERROR: Missing required key: {e}")
        return 2

    # Write only simulation_parameters block (clean)
    out_text = sim.to_text(indent=0, name="simulation_parameters")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(out_text)

    # Report
    print("=== Domain Constraint Fix Report ===")
    print(f"Input : {in_path}")
    print(f"Output: {out_path}")
    if changes:
        print("\nChanges made:")
        for c in changes:
            print(" -", c)
    else:
        print("\nNo changes were necessary (already satisfied all 3 constraints).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
