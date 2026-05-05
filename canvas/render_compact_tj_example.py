import argparse
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from render_stage2_summary_board_example import (
    BODY_FONT,
    CHIP_FONT,
    SMALL_FONT,
    CARD_TITLE_FONT,
    DATASET_DIRS,
    dataset_columns,
    detect_image_extension,
    detect_missing_regions,
    detect_missing_regions_from_reference,
    extract_final_answer,
    find_primary_problem_panel,
    get_font,
    highlight_image,
    line_height,
    map_boxes_to_reasoning,
    normalize_question,
    row_from_batch,
    thought_sequence,
    wrap_text,
)
import pyarrow.parquet as pq


OUTPUT_ROOT = Path(os.environ.get("UNIVLR_COMPACT_DEMO_DIR", "outputs/canvas/_compact_tj_demo"))
PATTERN_MAP = {
    "tjt": ("t", "j", "t"),
    "tjtt": ("t", "j", "t", "t"),
}
DATASET_ALIAS = {
    "visual_jigsaw": "2D Visual Reasoning - Visual Jigsaw",
    "visual_search": "2D Visual Reasoning - Visual Search",
    "maze": "Visual Logic & Strategic Games - Maze",
}
COMPACT_TITLE_FONT = get_font(22, bold=True)
COMPACT_CHIP_FONT = get_font(16, bold=True)
HIGHLIGHT_LABEL_FONT = get_font(15, bold=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one compact single-canvas example for (t,j,t) or (t,j,t,t), "
            "using a left joint-image column and a right text-step column."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_ALIAS),
        default="visual_search",
        help="Source dataset to scan.",
    )
    parser.add_argument(
        "--pattern",
        choices=sorted(PATTERN_MAP),
        default="tjtt",
        help="Target thought pattern.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        default=0,
        help="Which matched sample to render within the selected dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory used to save the rendered canvas and example json.",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=1536,
        help="Output canvas width.",
    )
    parser.add_argument(
        "--max-body-height",
        type=int,
        default=1180,
        help="Maximum height for the left joint-image panel.",
    )
    return parser.parse_args()


def resize_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    ratio = min(max_width / image.width, max_height / image.height)
    new_size = (max(1, int(round(image.width * ratio))), max(1, int(round(image.height * ratio))))
    if new_size == image.size:
        return image.copy()
    return image.resize(new_size, Image.LANCZOS)


def fit_text_to_box(text: str, max_width: int, max_height: int) -> tuple[ImageDraw.ImageDraw, object, list[str], int]:
    probe = Image.new("RGB", (max_width, max_height), (255, 255, 255))
    draw = ImageDraw.Draw(probe)
    for size in (36, 34, 32, 30, 28, 26, 24, 22, 20):
        font = get_font(size)
        lines = wrap_text(draw, text, font, max_width)
        lh = line_height(draw, font)
        total_h = len(lines) * lh + max(0, len(lines) - 1) * 8
        if total_h <= max_height:
            return draw, font, lines, lh
    font = get_font(20)
    lines = wrap_text(draw, text, font, max_width)
    lh = line_height(draw, font)
    max_lines = max(1, (max_height + 8) // max(1, lh + 8))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            last = lines[-1]
            if len(last) > 3:
                lines[-1] = last[:-3].rstrip() + "..."
    return draw, font, lines, lh


def compact_highlight_image(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    outline: tuple[int, int, int],
    fill_alpha: int = 40,
) -> Image.Image:
    if not boxes:
        return image.copy()
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, fill_alpha))
    highlighted = Image.alpha_composite(rgba, overlay)
    for box in boxes:
        highlighted.paste(rgba.crop(box), box)
    draw = ImageDraw.Draw(highlighted)
    for idx, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = box
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, outline=outline, width=4)
        badge_size = 26
        bx0 = x0 + 4
        by0 = max(4, y0 - 8)
        draw.rounded_rectangle((bx0, by0, bx0 + badge_size, by0 + badge_size), radius=9, fill=outline)
        text = str(idx)
        tb = draw.textbbox((0, 0), text, font=HIGHLIGHT_LABEL_FONT)
        tx = bx0 + (badge_size - (tb[2] - tb[0])) // 2
        ty = by0 + (badge_size - (tb[3] - tb[1])) // 2 - 1
        draw.text((tx, ty), text, fill=(255, 255, 255), font=HIGHLIGHT_LABEL_FONT)
    return highlighted.convert("RGB")


def draw_horizontal_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = (148, 163, 184),
) -> None:
    draw.line((start, end), fill=color, width=5)
    ex, ey = end
    size = 10
    if end[0] >= start[0]:
        head = [(ex, ey), (ex - size, ey - size), (ex - size, ey + size)]
    else:
        head = [(ex, ey), (ex + size, ey - size), (ex + size, ey + size)]
    draw.polygon(head, fill=color)


def draw_vertical_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = (148, 163, 184),
) -> None:
    draw.line((start, end), fill=color, width=5)
    ex, ey = end
    size = 10
    head = [(ex, ey), (ex - size, ey - size), (ex + size, ey - size)]
    draw.polygon(head, fill=color)


def render_text_card(step_label: str, body: str, width: int, height: int) -> Image.Image:
    card = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=26, fill=(255, 255, 255), outline=(191, 219, 254), width=3)
    draw.rounded_rectangle((0, 0, 14, height - 1), radius=26, fill=(59, 130, 246))

    padding_x = 28
    title_y = 22
    draw.text((padding_x, title_y), f"THOUGHT {step_label}", fill=(15, 23, 42), font=COMPACT_TITLE_FONT)
    chip_y = title_y + 36
    draw.rounded_rectangle((padding_x, chip_y, padding_x + 76, chip_y + 28), radius=12, fill=(219, 234, 254))
    draw.text((padding_x + 13, chip_y + 4), "TEXT", fill=(30, 64, 175), font=COMPACT_CHIP_FONT)

    body_y = chip_y + 44
    body_height = height - body_y - 22
    _, font, lines, lh = fit_text_to_box(body, width - padding_x * 2, body_height)
    cursor_y = body_y
    for line in lines:
        draw.text((padding_x, cursor_y), line, fill=(55, 65, 81), font=font)
        cursor_y += lh + 8
    return card


def render_joint_panel(step_label: str, body: str, image: Image.Image, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=26, fill=(255, 255, 255), outline=(187, 247, 208), width=3)
    draw.rounded_rectangle((0, 0, 14, height - 1), radius=26, fill=(34, 197, 94))

    padding = 24
    draw.text((padding, 22), f"THOUGHT {step_label}", fill=(15, 23, 42), font=COMPACT_TITLE_FONT)
    draw.rounded_rectangle((padding, 58, padding + 84, 86), radius=12, fill=(220, 252, 231))
    draw.text((padding + 11, 62), "JOINT", fill=(22, 101, 52), font=COMPACT_CHIP_FONT)

    text_y = 106
    text_max_h = min(240, max(110, int(height * 0.28)))
    _, font, lines, lh = fit_text_to_box(body, width - padding * 2, text_max_h)
    cursor_y = text_y
    for line in lines:
        draw.text((padding, cursor_y), line, fill=(55, 65, 81), font=font)
        cursor_y += lh + 8

    image_y = cursor_y + 14
    image_max_w = width - padding * 2
    image_max_h = height - image_y - padding
    fitted = resize_to_fit(image, image_max_w, image_max_h)
    img_x = (width - fitted.width) // 2
    img_y = image_y + max(0, (image_max_h - fitted.height) // 2)
    panel.paste(fitted, (img_x, img_y))
    return panel


def build_compact_canvas(
    highlighted_joint: Image.Image,
    joint_block: tuple[str, str],
    text_blocks: list[tuple[str, str, int]],
    thought_sequence: list[tuple[int, str]],
    canvas_width: int,
    max_body_height: int,
) -> Image.Image:
    margin = 36
    column_gap = 28
    header_h = 0
    footer_h = 0
    left_w = (canvas_width - margin * 2 - column_gap) // 2
    right_w = canvas_width - margin * 2 - column_gap - left_w

    joint_resized = resize_to_fit(highlighted_joint, left_w, max_body_height)
    _, joint_font, joint_lines, joint_lh = fit_text_to_box(joint_block[1], left_w - 48, 220)
    joint_text_h = len(joint_lines) * joint_lh + max(0, len(joint_lines) - 1) * 8
    body_h = joint_resized.height + joint_text_h + 190
    row_gap = 18
    row_count = len(text_blocks)
    row_h = max(180, (body_h - row_gap * max(0, row_count - 1)) // max(1, row_count))
    body_h = row_h * row_count + row_gap * max(0, row_count - 1)

    canvas_h = margin + header_h + body_h + footer_h + margin
    canvas = Image.new("RGB", (canvas_width, canvas_h), (246, 248, 252))
    draw = ImageDraw.Draw(canvas)

    body_y = margin + header_h
    joint_panel = render_joint_panel(joint_block[0], joint_block[1], joint_resized, left_w, body_h)
    canvas.paste(joint_panel, (margin, body_y))
    joint_box = (margin, body_y, margin + left_w, body_y + body_h)

    cursor_y = body_y
    card_boxes: dict[int, tuple[int, int, int, int]] = {}
    for idx, (thought_label, body, order_index) in enumerate(text_blocks):
        card = render_text_card(thought_label, body, right_w, row_h)
        card_x = margin + left_w + column_gap
        canvas.paste(card, (card_x, cursor_y))
        card_boxes[order_index] = (card_x, cursor_y, card_x + right_w, cursor_y + row_h)
        cursor_y += row_h
        if idx < len(text_blocks) - 1:
            cursor_y += row_gap
    sequence = sorted(thought_sequence, key=lambda item: item[0])
    for (src_idx, src_kind), (dst_idx, dst_kind) in zip(sequence, sequence[1:]):
        if src_kind == "t" and dst_kind == "j":
            src_box = card_boxes[src_idx]
            y = (src_box[1] + src_box[3]) // 2
            draw_horizontal_arrow(draw, (src_box[0] - 8, y), (joint_box[2] + 8, y))
        elif src_kind == "j" and dst_kind == "t":
            dst_box = card_boxes[dst_idx]
            y = (dst_box[1] + dst_box[3]) // 2
            draw_horizontal_arrow(draw, (joint_box[2] + 8, y), (dst_box[0] - 8, y))
        elif src_kind == "t" and dst_kind == "t":
            src_box = card_boxes[src_idx]
            dst_box = card_boxes[dst_idx]
            x = src_box[0] + (src_box[2] - src_box[0]) // 2
            draw_vertical_arrow(draw, (x, src_box[3] + 4), (x, dst_box[1] - 4))
    return canvas


def find_match(dataset_dir: Path, target_pattern: tuple[str, ...], match_index: int) -> tuple[dict, int]:
    columns = dataset_columns(dataset_dir)
    seen = 0
    source_sample_index = 0
    for parquet_path in sorted(dataset_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=128):
            for row_index in range(batch.num_rows):
                row = row_from_batch(batch, row_index, columns)
                blocks = thought_sequence(row["Text Reasoning Trace"] or "")
                pattern = tuple(block.kind for block in blocks)
                if pattern == target_pattern:
                    if seen == match_index:
                        row["_blocks"] = blocks
                        return row, source_sample_index
                    seen += 1
                source_sample_index += 1
    raise IndexError(f"Could not find match index {match_index} for pattern {''.join(target_pattern)} in {dataset_dir}")


def build_record(
    problem_path: str,
    canvas_path: str,
    question: str,
    final_answer: str,
    sample_id: str,
    dataset_name: str,
    source_sample_index: int,
    thought_pattern: list[str],
) -> dict:
    return {
        "metadata": {
            "dataset_name": "Zebra_CoT_compact_tj_demo",
            "sample_id": sample_id,
            "source_subset": dataset_name,
            "source_sample_index": source_sample_index,
            "thought_pattern": thought_pattern,
            "latent_canvas_count": 1,
            "latent_canvas_paths": [canvas_path],
            "problem_image_path": problem_path,
            "assistant_text_source": "Final Answer",
            "render_strategy": "compact_left_joint_right_text_v1",
        },
        "data": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": problem_path},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image", "image": canvas_path},
                    {"type": "text", "text": f"Therefore, the final asnwer is \\boxed{{{final_answer}}}."},
                ],
            },
        ],
    }


def main() -> None:
    args = parse_args()
    dataset_name = DATASET_ALIAS[args.dataset]
    dataset_dir = DATASET_DIRS[dataset_name]
    target_pattern = PATTERN_MAP[args.pattern]
    row, source_sample_index = find_match(dataset_dir, target_pattern, args.match_index)
    blocks = row["_blocks"]

    question = normalize_question(row["Question"] or "")
    final_answer = extract_final_answer(row["Final Answer"] or "")
    problem_bytes = row["problem_image_1"]["bytes"]
    problem_image = Image.open(io.BytesIO(problem_bytes)).convert("RGB")

    joint_block = next(block for block in blocks if block.kind == "j")
    joint_struct = row[joint_block.image_key]
    joint_image = Image.open(io.BytesIO(joint_struct["bytes"])).convert("RGB")
    primary_box = find_primary_problem_panel(problem_image)
    if primary_box is not None:
        missing_boxes = detect_missing_regions_from_reference(problem_image, primary_box, joint_image)
    else:
        missing_boxes = []
    if not missing_boxes:
        missing_boxes = detect_missing_regions(problem_image)
    mapped_boxes = map_boxes_to_reasoning(missing_boxes, primary_box, joint_image.size) if primary_box else []
    highlighted_joint = compact_highlight_image(joint_image, mapped_boxes, outline=(16, 185, 129), fill_alpha=36)

    if args.dataset == "visual_search" and joint_image.size != problem_image.size:
        missing_boxes = []
        mapped_boxes = []
        highlighted_joint = joint_image

    joint_block_tuple = (joint_block.index_label, joint_block.body)
    text_blocks = [
        (block.index_label, block.body, order_index)
        for order_index, block in enumerate(blocks)
        if block.kind == "t"
    ]
    canvas = build_compact_canvas(
        highlighted_joint=highlighted_joint,
        joint_block=joint_block_tuple,
        text_blocks=text_blocks,
        thought_sequence=[(order_index, block.kind) for order_index, block in enumerate(blocks)],
        canvas_width=args.canvas_width,
        max_body_height=args.max_body_height,
    )

    sample_dir = args.output_root / f"{args.dataset}_{args.pattern}_{args.match_index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    problem_path = sample_dir / f"problem{detect_image_extension(problem_bytes)}"
    canvas_path = sample_dir / "compact_canvas.png"
    example_path = sample_dir / "example.json"
    info_path = sample_dir / "render_info.json"

    problem_path.write_bytes(problem_bytes)
    canvas.save(canvas_path, compress_level=1)
    example = build_record(
        problem_path=str(problem_path),
        canvas_path=str(canvas_path),
        question=question,
        final_answer=final_answer,
        sample_id=sample_dir.name,
        dataset_name=dataset_name,
        source_sample_index=source_sample_index,
        thought_pattern=[block.kind for block in blocks],
    )
    example_path.write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    info_path.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "pattern": args.pattern,
                "match_index": args.match_index,
                "source_sample_index": source_sample_index,
                "question": question,
                "final_answer": final_answer,
                "missing_regions": missing_boxes,
                "mapped_boxes": mapped_boxes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved compact canvas to {canvas_path}")
    print(f"Saved example json to {example_path}")


if __name__ == "__main__":
    main()
