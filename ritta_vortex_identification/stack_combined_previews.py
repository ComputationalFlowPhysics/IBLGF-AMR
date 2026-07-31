"""Stack labeled beginning/middle/end preview rows into one PNG."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def stack_rows(rows, output_path, font_size=56, header_height=90, row_gap=12):
    """Place a centered label above each image and stack the rows vertically."""
    loaded_rows = []
    try:
        for label, image_path in rows:
            with Image.open(image_path) as image:
                loaded_rows.append((label, image.convert("RGB")))

        width = max(image.width for _, image in loaded_rows)
        height = sum(header_height + image.height for _, image in loaded_rows)
        height += row_gap * (len(loaded_rows) - 1)
        canvas = Image.new("RGB", (width, height), "white")
        drawing = ImageDraw.Draw(canvas)
        font = load_font(font_size)

        y_offset = 0
        for label, image in loaded_rows:
            bounds = drawing.textbbox((0, 0), label, font=font)
            text_width = bounds[2] - bounds[0]
            text_height = bounds[3] - bounds[1]
            text_x = (width - text_width) // 2
            text_y = y_offset + (header_height - text_height) // 2 - bounds[1]
            drawing.text((text_x, text_y), label, fill="black", font=font)

            image_x = (width - image.width) // 2
            canvas.paste(image, (image_x, y_offset + header_height))
            y_offset += header_height + image.height + row_gap

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        canvas.close()
    finally:
        for _, image in loaded_rows:
            image.close()


def main():
    parser = argparse.ArgumentParser(
        description="Vertically stack labeled combined-preview PNG rows."
    )
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--row",
        action="append",
        nargs=2,
        required=True,
        metavar=("LABEL", "IMAGE"),
        help="Row label and combined-preview PNG; repeat once per row.",
    )
    parser.add_argument("--font-size", type=int, default=56)
    parser.add_argument("--header-height", type=int, default=90)
    parser.add_argument("--row-gap", type=int, default=12)
    args = parser.parse_args()

    if args.font_size <= 0:
        parser.error("--font-size must be positive")
    if args.header_height <= 0:
        parser.error("--header-height must be positive")
    if args.row_gap < 0:
        parser.error("--row-gap must be non-negative")

    rows = []
    for label, image_name in args.row:
        image_path = Path(image_name).expanduser().resolve()
        if not image_path.is_file():
            parser.error("Row image does not exist: {}".format(image_path))
        rows.append((label, image_path))

    output_path = args.output_file.expanduser().resolve()
    stack_rows(
        rows,
        output_path,
        font_size=args.font_size,
        header_height=args.header_height,
        row_gap=args.row_gap,
    )
    print("Saved {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
