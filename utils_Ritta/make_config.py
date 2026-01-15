#!/usr/bin/env python3
"""
make_config.py

Interactive generator/editor for IBLGF-style config files.
Constraint enforced:
  For each dimension, nBlocks = bd_extent / block_extent must be an integer power of 2.
If violated, bd_extent is adjusted to the closest valid value and the user is notified.

Supports:
  - Create from scratch (prompts all parameters)
  - Create based on existing config (parses nested blocks)
  - Edit via parameter paths (e.g., cfl, domain.bd_extent, output.directory)

No external dependencies.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, List, Tuple, Union, Optional
import readline
import glob

# ----------------------------
# Parsing / Formatting helpers
# ----------------------------

Number = Union[int, float]
Value = Union[Number, bool, str, Tuple[Any, ...]]

def path_completer(text, state):
    text = os.path.expanduser(text)

    if not text:
        matches = glob.glob("*")
    else:
        matches = glob.glob(text + "*")

    matches = [m + "/" if os.path.isdir(m) else m for m in matches]
    matches.sort()

    try:
        return matches[state]
    except IndexError:
        return None

readline.set_completer(path_completer)

# IMPORTANT: macOS often uses libedit, which needs different bindings
if "libedit" in (readline.__doc__ or "").lower():
    # bind TAB (Ctrl+I) to completion in libedit
    readline.parse_and_bind("bind ^I rl_complete")
    # Optional: list options on double-TAB usually works by default in libedit
else:
    # GNU readline
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set show-all-if-ambiguous on")
    readline.parse_and_bind("TAB: menu-complete")

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def nearest_power_of_two_int(x: float) -> int:
    """
    Returns the nearest power-of-two integer to x.
    Ties go to the larger power (arbitrary but predictable).
    """
    if x <= 1:
        return 1
    # Find surrounding powers of 2
    lo = 1
    while lo * 2 <= x:
        lo *= 2
    hi = lo * 2
    # Choose closer
    if abs(x - lo) < abs(hi - x):
        return lo
    else:
        return hi


def parse_tuple(s: str) -> Tuple[Any, ...]:
    # s like "(96,96,96)" possibly with spaces
    inner = s.strip()[1:-1].strip()
    if not inner:
        return tuple()
    parts = [p.strip() for p in inner.split(",")]
    return tuple(parse_scalar(p) for p in parts)


def parse_scalar(s: str) -> Value:
    s = s.strip()
    # tuple
    if s.startswith("(") and s.endswith(")"):
        return parse_tuple(s)

    # bool
    if s == "true":
        return True
    if s == "false":
        return False

    # quoted string (rare here but we support)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    # bare string (e.g. output directory, hdf5_ref_name)
    # but could also be number in scientific notation
    # try int, float in that order
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        # float / sci
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", s):
            return float(s)
    except Exception:
        pass

    return s


def format_value(v: Value) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, tuple):
        return "(" + ",".join(format_value(x) for x in v) + ")"
    if isinstance(v, (int, float)):
        # Keep scientific for very small/large, otherwise compact
        if isinstance(v, float):
            # Avoid printing 1.0 as 1.0? Either is fine; keep compact.
            return f"{v:.15g}"
        return str(v)
    # string
    return str(v)


@dataclass
class Block:
    """
    A config block is an ordered mapping of keys -> values or nested blocks.
    """
    items: "OrderedDict[str, Any]"

    def __init__(self) -> None:
        self.items = OrderedDict()

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
                # auto-create blocks if needed
                cur.items[p] = Block()
            cur = cur.items[p]
        cur.items[parts[-1]] = value

    def list_paths(self, prefix: str = "") -> List[str]:
        out: List[str] = []
        for k, v in self.items.items():
            p = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, Block):
                out.extend(v.list_paths(p))
            else:
                out.append(p)
        return out

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
    """
    Tokenize into meaningful tokens:
      - identifiers
      - '{', '}', '=', ';'
      - tuples '(...)' as one token
      - barewords/numbers (until delimiter)
    Also strips comments //...
    """
    # remove // comments
    no_comments = re.sub(r"//.*", "", text)

    tokens: List[str] = []
    i = 0
    n = len(no_comments)

    def skip_ws(idx: int) -> int:
        while idx < n and no_comments[idx].isspace():
            idx += 1
        return idx

    while i < n:
        i = skip_ws(i)
        if i >= n:
            break
        ch = no_comments[i]

        if ch in "{}=;":
            tokens.append(ch)
            i += 1
            continue

        # tuple literal
        if ch == "(":
            depth = 0
            j = i
            while j < n:
                if no_comments[j] == "(":
                    depth += 1
                elif no_comments[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            tokens.append(no_comments[i:j].strip())
            i = j
            continue

        # identifier / number / bareword
        j = i
        while j < n and (not no_comments[j].isspace()) and no_comments[j] not in "{}=;":
            j += 1
        tokens.append(no_comments[i:j].strip())
        i = j

    # remove empties
    return [t for t in tokens if t]


def parse_config(text: str) -> Block:
    """
    Parses text like:
      simulation_parameters { a=1; block { x=2; } }

    Returns a Block whose top-level keys include 'simulation_parameters' as a Block.
    """
    toks = tokenize_config(text)
    pos = 0

    def expect(tok: str) -> None:
        nonlocal pos
        if pos >= len(toks) or toks[pos] != tok:
            got = toks[pos] if pos < len(toks) else "<EOF>"
            raise ValueError(f"Expected '{tok}' but got '{got}' at token index {pos}.")
        pos += 1

    def parse_block_contents() -> Block:
        nonlocal pos
        b = Block()
        while pos < len(toks) and toks[pos] != "}":
            # Could be: key '=' value ';'  OR  key '{' ... '}'
            key = toks[pos]
            pos += 1
            if pos < len(toks) and toks[pos] == "{":
                pos += 1
                child = parse_block_contents()
                expect("}")
                b.items[key] = child
            else:
                expect("=")
                if pos >= len(toks):
                    raise ValueError("Unexpected EOF while reading value.")
                val_tok = toks[pos]
                pos += 1
                val = parse_scalar(val_tok)
                expect(";")
                b.items[key] = val
        return b

    root = Block()
    while pos < len(toks):
        name = toks[pos]
        pos += 1
        expect("{")
        b = parse_block_contents()
        expect("}")
        root.items[name] = b

    return root


# ----------------------------
# Constraint enforcement
# ----------------------------

def coerce_to_3tuple(v: Value) -> Tuple[int, int, int]:
    if isinstance(v, tuple) and len(v) == 3:
        return (int(v[0]), int(v[1]), int(v[2]))
    if isinstance(v, (int, float)):
        x = int(v)
        return (x, x, x)
    raise ValueError(f"Expected a 3-tuple or scalar, got: {v!r}")


def enforce_domain_block_constraint(sim: Block) -> list[str]:
    """
    Enforces:
      1) bd_extent / block_extent is an integer power of 2 (per dimension)
      2) domain.block.extent == domain.bd_extent
      3) domain.block.base   == domain.bd_base
    """
    msgs: list[str] = []

    # ---- required paths ----
    try:
        bd_base   = sim.get_path("domain.bd_base")
        bd_extent = sim.get_path("domain.bd_extent")
        block_ext = sim.get_path("domain.block_extent")
    except KeyError:
        return msgs  # domain not fully specified

    # normalize tuples (2D or 3D supported)
    def as_tuple(v):
        if isinstance(v, tuple):
            return tuple(int(x) for x in v)
        return (int(v),) * 3

    bd_base   = as_tuple(bd_base)
    bd_extent = as_tuple(bd_extent)
    block_ext = int(block_ext)

    # ---- 1) enforce power-of-two blocks ----
    new_extent = list(bd_extent)
    changed_extent = False

    for d in range(len(bd_extent)):
        n_blocks = bd_extent[d] / block_ext
        if not (n_blocks.is_integer() and is_power_of_two(int(n_blocks))):
            target_blocks = nearest_power_of_two_int(n_blocks)
            fixed = target_blocks * block_ext
            msgs.append(
                f"Adjusted domain.bd_extent[{d}] "
                f"from {bd_extent[d]} → {fixed} "
                f"to satisfy power-of-two block constraint."
            )
            new_extent[d] = fixed
            changed_extent = True

    if changed_extent:
        bd_extent = tuple(new_extent)
        sim.set_path("domain.bd_extent", bd_extent)

    # ---- 2) sync block.extent ----
    try:
        sim.get_path("domain.block.extent")
        sim.set_path("domain.block.extent", bd_extent)
        msgs.append("Synced domain.block.extent to domain.bd_extent.")
    except KeyError:
        pass

    # ---- 3) sync block.base ----
    try:
        sim.get_path("domain.block.base")
        sim.set_path("domain.block.base", bd_base)
        msgs.append("Synced domain.block.base to domain.bd_base.")
    except KeyError:
        pass

    return msgs


# ----------------------------
# Interactive prompting
# ----------------------------

DEFAULT_TEMPLATE = """
simulation_parameters
{
    nLevels=0;
    Ux=0.0;
    // Time Marching
    source_max=13.0;
    nBaseLevelTimeSteps=4;
    cfl = 0.35;
    cfl_max = 1000;
    Re = 1000.0;
    refinement_factor=0.125;
    R=0.5;
    DistanceOfVortexRings=0.25;
    adapt_frequency=10;
    output_frequency=4;
    base_level_threshold=1e-4;
    hard_max_refinement=true;

    single_ring=true;
    perturbation=false;
    vDelta=0.2;
    Vort_type=1;
    fat_ring=true;

    //hdf5_ic_name=ic.hdf5;
    hdf5_ref_name=ref.hdf5;

    output
    {
        directory=output;
    }

    restart_write_frequency=20;
    write_restart=false;
    use_restart=false;

    restart
    {
        load_directory=restart;
        save_directory=restart;
    }

    domain
    {
        bd_base = (-48,-48,-48);
        bd_extent = (96,96,96);

        dx_base=0.06125;

        block_extent=6;

        block
        {
            base = (-48,-48,-48);
            extent = (96,96,96);
        }
    }
    EXP_LInf=1e-3;
}
""".strip() + "\n"


def prompt_yes_no(msg: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{msg} [{d}]: ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def prompt_string(msg: str, default: Optional[str] = None) -> str:
    if default is None:
        ans = input(f"{msg}: ").strip()
        return ans
    ans = input(f"{msg} [default: {default}]: ").strip()
    return ans if ans else default


def prompt_value(msg: str, default: Value) -> Value:
    """
    Prompts for a value; user can enter:
      - numbers (int/float/sci)
      - true/false
      - tuples like (1,2,3)
      - bare string
    """
    d = format_value(default)
    ans = input(f"{msg} [default: {d}]: ").strip()
    if not ans:
        return default
    return parse_scalar(ans)


def build_from_scratch() -> Block:
    # Use the template as the starting point, then prompt over all scalar leaves.
    root = parse_config(DEFAULT_TEMPLATE)
    sim: Block = root.items["simulation_parameters"]

    print("\nCreating from scratch. Press Enter to keep defaults.\n")

    # Prompt all scalar paths in stable order
    for path in sim.list_paths():
        cur = sim.get_path(path)
        # skip block "domain.block.extent/base"? No, include them (but they'll get synced)
        new_val = prompt_value(f"Set {path}", cur)
        sim.set_path(path, new_val)

        # If user sets domain.block.extent/base directly, we don't enforce anything yet.
        # We'll enforce after.

    # Enforce constraint
    msgs = enforce_domain_block_constraint(sim)
    for m in msgs:
        print("NOTE:", m)

    return root


def load_based_on_existing() -> Block:
    print("\nTip: Use TAB to auto-complete file paths.\n")

    while True:
        base_path = prompt_string(
            "Path to existing config to base on (e.g., ./configFile_0)"
        )

        # Expand ~ and env vars
        base_path = os.path.expandvars(os.path.expanduser(base_path))

        if os.path.isfile(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                text = f.read()
            try:
                root = parse_config(text)
                if "simulation_parameters" not in root.items:
                    print("That file parsed, but didn't contain a 'simulation_parameters { ... }' block.")
                    continue
                return root
            except Exception as e:
                print(f"Failed to parse config: {e}")
        else:
            print("File not found. Try again.")


def edit_loop(root: Block) -> None:
    sim: Block = root.items["simulation_parameters"]
    print("\nEditing mode. Type parameter names like:")
    print("  cfl")
    print("  domain.bd_extent")
    print("  domain.block_extent")
    print("  output.directory")
    print("Parameter inside another parameter must be entered in format parent.child")
    print("You can also type 'list' to see all parameter paths.\n")

    while True:
        key = input("Parameter to change (or 'list'): ").strip()
        if not key:
            continue
        if key.lower() == "list":
            for p in sim.list_paths():
                print(" ", p)
            continue

        # Accept bare keys at top-level (like "cfl") and also nested keys.
        try:
            cur = sim.get_path(key)
        except KeyError as e:
            print(f"Not found: {e}")
            continue

        new_val = prompt_value(f"New value for {key}", cur)
        sim.set_path(key, new_val)

        # If domain/block_extent changed OR bd_extent changed, enforce constraint
        if key in ("domain.block_extent", "domain.bd_extent"):
            msgs = enforce_domain_block_constraint(sim)
            if msgs:
                for m in msgs:
                    print("NOTE:", m)
            else:
                print("NOTE: No domain adjustment needed (already valid).")

        more = prompt_yes_no("Anything else to change?", default=True)
        if not more:
            break


def write_output(root: Block) -> None:
    out_path = prompt_string(
        "Output file path OR directory for the NEW config (e.g., ./configFile_new or ./configs/)"
    ).strip()

    # Expand ~ and env vars
    out_path = os.path.expandvars(os.path.expanduser(out_path))

    # If user gave a directory (or ended with /), choose a default filename inside it
    if out_path.endswith(os.sep) or (os.path.exists(out_path) and os.path.isdir(out_path)):
        out_dir = out_path.rstrip(os.sep) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "configFile_new")
        print(f"NOTE: You entered a directory. Writing to: {out_path}")
    else:
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)

    # Render only the simulation_parameters block
    text = root.items["simulation_parameters"].to_text(indent=0, name="simulation_parameters")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    make_exec = prompt_yes_no(
        "Make the output file executable (Finder may show it as 'Unix 실행 파일')?", default=True
    )
    if make_exec:
        st = os.stat(out_path)
        os.chmod(out_path, st.st_mode | 0o111)

    print(f"\nWrote new config to: {out_path}")
    if make_exec:
        print("Marked as executable.")
    print("Done.")


def main() -> int:
    print("=== Config Builder (enforces: bd_extent/block_extent is power-of-2) ===\n")

    from_scratch = prompt_yes_no("Create a new config FROM SCRATCH?", default=False)

    if from_scratch:
        root = build_from_scratch()
    else:
        root = load_based_on_existing()
        sim: Block = root.items["simulation_parameters"]
        # Enforce once on load too (in case base file violates)
        msgs = enforce_domain_block_constraint(sim)
        for m in msgs:
            print("NOTE:", m)

        edit_loop(root)

    write_output(root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        raise SystemExit(1)
