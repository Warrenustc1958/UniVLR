import argparse
import io
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


DATA_ROOT = Path(os.environ.get("UNIVLR_DATA_ROOT", "data"))
STAGE2_JSONL = Path(
    os.environ.get("UNIVLR_STAGE2_JSONL", str(DATA_ROOT / "Zebra_CoT_visual_search/stage2_train.jsonl"))
)
OUTPUT_ROOT = Path(os.environ.get("UNIVLR_SUMMARY_BOARD_DEMO_DIR", "outputs/canvas/_summary_board_demo"))
DATASET_DIRS = {
    "2D Visual Reasoning - Visual Search": Path(
        os.environ.get(
            "UNIVLR_VISUAL_SEARCH_DIR",
            str(DATA_ROOT / "zebra-cot/2D Visual Reasoning - Visual Search"),
        )
    ),
    "2D Visual Reasoning - Visual Jigsaw": Path(
        os.environ.get(
            "UNIVLR_VISUAL_JIGSAW_DIR",
            str(DATA_ROOT / "zebra-cot/2D Visual Reasoning - Visual Jigsaw"),
        )
    ),
    "Visual Logic & Strategic Games - Maze": Path(
        os.environ.get(
            "UNIVLR_MAZE_DIR",
            str(DATA_ROOT / "zebra-cot/Visual Logic & Strategic Games - Maze"),
        )
    ),
}
THOUGHT_PATTERN = re.compile(r"THOUGHT\s+([0-9N]+):")
IMAGE_TOKEN_PATTERN = re.compile(r"<image_start>\[(reasoning_image_\d+)\]<image_end>")
PROBLEM_IMAGE_TOKEN_PATTERN = re.compile(r"\s*<image_start>\[problem_image_1\]<image_end>\s*")


@dataclass
class ThoughtBlock:
    index_label: str
    body: str
    kind: str
    image_key: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one stage2 sample into a single large multi-step summary board "
            "with arrows and local focus highlights."
        )
    )
    parser.add_argument(
        "--stage2-jsonl",
        type=Path,
        default=STAGE2_JSONL,
        help="Existing stage2 jsonl used to locate the source sample.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=0,
        help="Sample id inside the stage2 jsonl to render.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory used to save the rendered board and example json.",
    )
    parser.add_argument(
        "--board-width",
        type=int,
        default=1792,
        help="Target width for the summary board canvas.",
    )
    parser.add_argument(
        "--show-final-answer",
        action="store_true",
        help="Also draw the final answer badge on the board image.",
    )
    return parser.parse_args()


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


DISPLAY_FONT = get_font(52, bold=True)
SUBTITLE_FONT = get_font(24)
SECTION_FONT = get_font(30, bold=True)
CARD_TITLE_FONT = get_font(28, bold=True)
CHIP_FONT = get_font(22, bold=True)
BODY_FONT = get_font(25)
SMALL_FONT = get_font(20)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1]


def sort_reasoning_columns(columns: Iterable[str]) -> list[str]:
    def order_key(name: str) -> int:
        suffix = name.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 10**9

    return sorted((c for c in columns if c.startswith("reasoning_image_")), key=order_key)


def dataset_columns(dataset_dir: Path) -> list[str]:
    first_file = sorted(dataset_dir.glob("*.parquet"))[0]
    schema_names = pq.ParquetFile(first_file).schema_arrow.names
    reasoning_columns = sort_reasoning_columns(schema_names)
    return ["Question", "Text Reasoning Trace", "Final Answer", "problem_image_1", *reasoning_columns]


def row_from_batch(batch, row_index: int, columns: list[str]) -> dict:
    row = {}
    for name in columns:
        row[name] = batch.column(batch.schema.get_field_index(name))[row_index].as_py()
    return row


def load_stage2_record(stage2_jsonl: Path, sample_id: int) -> dict:
    with stage2_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("metadata", {}).get("sample_id") == sample_id:
                return record
    raise ValueError(f"Sample id {sample_id} was not found in {stage2_jsonl}.")


def fetch_source_row(dataset_dir: Path, source_index: int) -> dict:
    columns = dataset_columns(dataset_dir)
    current = 0
    for parquet_path in sorted(dataset_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=128):
            if current + batch.num_rows <= source_index:
                current += batch.num_rows
                continue
            row_index = source_index - current
            return row_from_batch(batch, row_index, columns)
    raise IndexError(f"Source sample index {source_index} is out of range for {dataset_dir}.")


def normalize_question(question: str) -> str:
    return PROBLEM_IMAGE_TOKEN_PATTERN.sub("", question).strip()


def extract_final_answer(text: str) -> str:
    answer = text.strip()
    answer = re.sub(r"^\s*(?:the answer is|answer)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE)
    return answer.rstrip(" .")


def thought_sequence(trace: str) -> list[ThoughtBlock]:
    matches = list(THOUGHT_PATTERN.finditer(trace))
    blocks: list[ThoughtBlock] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(trace)
        raw_block = trace[start:end].strip()
        image_match = IMAGE_TOKEN_PATTERN.search(raw_block)
        body = IMAGE_TOKEN_PATTERN.sub("", raw_block).strip()
        blocks.append(
            ThoughtBlock(
                index_label=match.group(1),
                body=body,
                kind="j" if image_match else "t",
                image_key=image_match.group(1) if image_match else None,
            )
        )
    return blocks


def detect_image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def resized_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.copy()
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.LANCZOS)


def expand_box(box: tuple[int, int, int, int], width: int, height: int, scale: float = 1.25) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    half_w = (x1 - x0) * scale / 2
    half_h = (y1 - y0) * scale / 2
    nx0 = max(0, int(round(cx - half_w)))
    ny0 = max(0, int(round(cy - half_h)))
    nx1 = min(width, int(round(cx + half_w)))
    ny1 = min(height, int(round(cy + half_h)))
    return nx0, ny0, nx1, ny1


def connected_components_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while queue:
                cx, cy = queue.popleft()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            if area >= min_area:
                boxes.append((min_x, min_y, max_x + 1, max_y + 1))
    boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    return boxes


def merge_close_boxes(
    boxes: list[tuple[int, int, int, int]],
    max_gap: int = 28,
) -> list[tuple[int, int, int, int]]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int]] = []
        while merged:
            ax0, ay0, ax1, ay1 = merged.pop(0)
            current = (ax0, ay0, ax1, ay1)
            keep_searching = True
            while keep_searching:
                keep_searching = False
                for idx, box in enumerate(merged):
                    bx0, by0, bx1, by1 = box
                    if ax0 <= bx1 + max_gap and bx0 <= ax1 + max_gap and ay0 <= by1 + max_gap and by0 <= ay1 + max_gap:
                        ax0 = min(ax0, bx0)
                        ay0 = min(ay0, by0)
                        ax1 = max(ax1, bx1)
                        ay1 = max(ay1, by1)
                        current = (ax0, ay0, ax1, ay1)
                        merged.pop(idx)
                        changed = True
                        keep_searching = True
                        break
            next_boxes.append(current)
        merged = next_boxes
    merged.sort(key=lambda box: (box[1], box[0]))
    return merged


def detect_missing_regions_from_reference(
    problem_image: Image.Image,
    primary_box: tuple[int, int, int, int],
    reference_image: Image.Image,
    max_regions: int = 4,
) -> list[tuple[int, int, int, int]]:
    px0, py0, px1, py1 = primary_box
    if px1 - px0 < 40 or py1 - py0 < 40:
        return []
    crop = problem_image.crop(primary_box).resize(reference_image.size, Image.LANCZOS).convert("RGB")
    crop_arr = np.asarray(crop).astype(np.int16)
    ref_arr = np.asarray(reference_image.convert("RGB")).astype(np.int16)
    diff = np.abs(crop_arr - ref_arr).mean(axis=2)
    channel_range = crop_arr.max(axis=2) - crop_arr.min(axis=2)
    gray_like = channel_range <= 28
    mask = (diff >= 34) & gray_like
    min_area = max(60, int(mask.size * 0.001))
    ref_boxes = connected_components_boxes(mask, min_area=min_area)
    mapped: list[tuple[int, int, int, int]] = []
    ref_w, ref_h = reference_image.size
    box_scale_x = (px1 - px0) / max(1, ref_w)
    box_scale_y = (py1 - py0) / max(1, ref_h)
    for box in ref_boxes[: max_regions * 2]:
        x0, y0, x1, y1 = box
        mapped.append(
            (
                int(round(px0 + x0 * box_scale_x)),
                int(round(py0 + y0 * box_scale_y)),
                int(round(px0 + x1 * box_scale_x)),
                int(round(py0 + y1 * box_scale_y)),
            )
        )
    mapped = merge_close_boxes(mapped)
    return mapped[:max_regions]


def detect_missing_regions(problem_image: Image.Image, max_regions: int = 4) -> list[tuple[int, int, int, int]]:
    scale = max(problem_image.width, problem_image.height) / 900
    if scale < 1:
        scale = 1
    small = problem_image.resize(
        (max(1, int(round(problem_image.width / scale))), max(1, int(round(problem_image.height / scale)))),
        Image.NEAREST,
    ).convert("RGB")
    arr = np.asarray(small).astype(np.int16)
    channel_range = arr.max(axis=2) - arr.min(axis=2)
    mean = arr.mean(axis=2)
    gray_mask = (channel_range <= 14) & (mean >= 90) & (mean <= 225)
    min_area = max(40, int(gray_mask.size * 0.0012))
    small_boxes = connected_components_boxes(gray_mask, min_area=min_area)
    scaled_boxes: list[tuple[int, int, int, int]] = []
    for box in small_boxes[: max_regions * 2]:
        x0, y0, x1, y1 = box
        scaled_boxes.append(
            (
                int(round(x0 * scale)),
                int(round(y0 * scale)),
                int(round(x1 * scale)),
                int(round(y1 * scale)),
            )
        )
    scaled_boxes = merge_close_boxes(scaled_boxes)
    return scaled_boxes[:max_regions]


def find_primary_problem_panel(problem_image: Image.Image) -> tuple[int, int, int, int] | None:
    arr = np.asarray(problem_image.convert("RGB")).astype(np.int16)
    mean = arr.mean(axis=2)
    cutoff = int(arr.shape[0] * 0.72)
    mask = mean[:cutoff] < 245
    row_density = mask.mean(axis=1)
    active_rows = row_density > 0.22
    best_start = best_end = None
    start = None
    for idx, active in enumerate(active_rows):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if best_start is None or idx - start > best_end - best_start:
                best_start, best_end = start, idx
            start = None
    if start is not None and (best_start is None or len(active_rows) - start > best_end - best_start):
        best_start, best_end = start, len(active_rows)
    if best_start is None or best_end - best_start < 40:
        return None
    col_density = mask[best_start:best_end].mean(axis=0)
    active_cols = col_density > 0.55
    col_indices = np.flatnonzero(active_cols)
    if len(col_indices) < 40:
        return None
    return int(col_indices[0]), int(best_start), int(col_indices[-1] + 1), int(best_end)


def map_boxes_to_reasoning(
    boxes: list[tuple[int, int, int, int]],
    primary_box: tuple[int, int, int, int] | None,
    reasoning_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    if primary_box is None:
        return []
    px0, py0, px1, py1 = primary_box
    panel_w = max(1, px1 - px0)
    panel_h = max(1, py1 - py0)
    rw, rh = reasoning_size
    mapped = []
    for box in boxes:
        x0, y0, x1, y1 = box
        ix0 = max(px0, min(px1, x0))
        iy0 = max(py0, min(py1, y0))
        ix1 = max(px0, min(px1, x1))
        iy1 = max(py0, min(py1, y1))
        if ix1 - ix0 < 8 or iy1 - iy0 < 8:
            continue
        mapped.append(
            (
                int(round((ix0 - px0) / panel_w * rw)),
                int(round((iy0 - py0) / panel_h * rh)),
                int(round((ix1 - px0) / panel_w * rw)),
                int(round((iy1 - py0) / panel_h * rh)),
            )
        )
    return mapped


def highlight_image(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    outline: tuple[int, int, int],
    fill_alpha: int = 120,
) -> Image.Image:
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, fill_alpha))
    dimmed = Image.alpha_composite(rgba, overlay)
    highlighted = dimmed.copy()
    for box in boxes:
        crop = rgba.crop(box)
        highlighted.paste(crop, box)
    draw = ImageDraw.Draw(highlighted)
    for idx, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = box
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, outline=outline, width=8)
        badge_w = 42
        badge_h = 42
        bx0 = x0 + 8
        by0 = max(8, y0 - 18)
        draw.rounded_rectangle((bx0, by0, bx0 + badge_w, by0 + badge_h), radius=14, fill=outline)
        draw.text((bx0 + 13, by0 + 6), str(idx), fill=(255, 255, 255), font=CHIP_FONT)
    return highlighted.convert("RGB")


def render_zoom_chips(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    max_width: int,
) -> Image.Image | None:
    if not boxes:
        return None
    chip_size = 146
    gap = 18
    chips = []
    for idx, box in enumerate(boxes[:3], start=1):
        crop_box = expand_box(box, image.width, image.height, scale=1.5)
        crop = image.crop(crop_box).resize((chip_size, chip_size), Image.LANCZOS)
        chip = Image.new("RGB", (chip_size, chip_size + 42), (255, 255, 255))
        chip_draw = ImageDraw.Draw(chip)
        chip.paste(crop, (0, 42))
        chip_draw.rounded_rectangle((0, 42, chip_size - 1, chip_size + 41), radius=18, outline=(251, 191, 36), width=4)
        chip_draw.rounded_rectangle((0, 0, 44, 34), radius=12, fill=(251, 191, 36))
        chip_draw.text((15, 4), str(idx), fill=(255, 255, 255), font=CHIP_FONT)
        chips.append(chip)
    width = len(chips) * chip_size + max(0, len(chips) - 1) * gap
    if width > max_width:
        return None
    strip = Image.new("RGB", (width, chip_size + 42), (255, 255, 255))
    cursor_x = 0
    for chip in chips:
        strip.paste(chip, (cursor_x, 0))
        cursor_x += chip_size + gap
    return strip


def render_problem_panel(
    image: Image.Image,
    question: str,
    boxes: list[tuple[int, int, int, int]],
    panel_width: int,
) -> Image.Image:
    padding = 30
    inner_width = panel_width - padding * 2
    question_lines = wrap_text(ImageDraw.Draw(Image.new("RGB", (10, 10))), question, SMALL_FONT, inner_width)
    highlighted = highlight_image(image, boxes, outline=(251, 191, 36))
    highlighted = resized_to_width(highlighted, inner_width)
    chip_strip = render_zoom_chips(image, boxes, inner_width)
    text_height = len(question_lines) * 26 + 18
    chip_height = chip_strip.height + 24 if chip_strip is not None else 0
    height = padding + 42 + 22 + text_height + 22 + highlighted.height + chip_height + padding
    panel = Image.new("RGB", (panel_width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, panel_width - 1, height - 1), radius=28, fill=(255, 255, 255), outline=(226, 232, 240), width=3)
    draw.text((padding, padding), "Problem + Focus Regions", fill=(15, 23, 42), font=SECTION_FONT)
    cursor_y = padding + 48
    for line in question_lines[:4]:
        draw.text((padding, cursor_y), line, fill=(71, 85, 105), font=SMALL_FONT)
        cursor_y += 26
    cursor_y += 14
    panel.paste(highlighted, (padding, cursor_y))
    cursor_y += highlighted.height + 24
    if chip_strip is not None:
        panel.paste(chip_strip, (padding, cursor_y))
    return panel


def render_reasoning_panel(
    image: Image.Image,
    mapped_boxes: list[tuple[int, int, int, int]],
    max_width: int,
) -> Image.Image:
    if mapped_boxes:
        rendered = highlight_image(image, mapped_boxes, outline=(16, 185, 129), fill_alpha=70)
    else:
        rendered = image.copy()
    return resized_to_width(rendered, max_width)


def render_step_card(
    step_number: int,
    block: ThoughtBlock,
    card_width: int,
    reasoning_image: Image.Image | None = None,
    mapped_boxes: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    padding = 24
    inner_width = card_width - padding * 2 - 12
    draft = Image.new("RGB", (card_width, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(draft)
    lines = wrap_text(draw, block.body, BODY_FONT, inner_width)
    title_h = line_height(draw, CARD_TITLE_FONT)
    body_h = len(lines) * 32 + max(0, len(lines) - 1) * 6
    image_panel = None
    image_panel_h = 0
    if reasoning_image is not None:
        image_panel = render_reasoning_panel(reasoning_image, mapped_boxes or [], inner_width)
        image_panel_h = image_panel.height + 20
    height = padding + title_h + 20 + 36 + 22 + body_h + image_panel_h + padding
    card = Image.new("RGB", (card_width, height), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    if block.kind == "t":
        accent = (59, 130, 246)
        border = (191, 219, 254)
        chip_fill = (219, 234, 254)
        chip_text = (30, 64, 175)
        chip_label = "TEXT"
    else:
        accent = (34, 197, 94)
        border = (187, 247, 208)
        chip_fill = (220, 252, 231)
        chip_text = (22, 101, 52)
        chip_label = "JOINT"

    draw.rounded_rectangle((0, 0, card_width - 1, height - 1), radius=26, fill=(255, 255, 255), outline=border, width=3)
    draw.rounded_rectangle((0, 0, 16, height - 1), radius=26, fill=accent)
    draw.text((padding + 4, padding), f"STEP {step_number}", fill=(15, 23, 42), font=CARD_TITLE_FONT)
    cursor_y = padding + title_h + 16
    chip_bbox = draw.textbbox((0, 0), chip_label, font=CHIP_FONT)
    chip_w = chip_bbox[2] - chip_bbox[0] + 26
    draw.rounded_rectangle((padding + 4, cursor_y, padding + 4 + chip_w, cursor_y + 36), radius=16, fill=chip_fill)
    draw.text((padding + 17, cursor_y + 6), chip_label, fill=chip_text, font=CHIP_FONT)
    cursor_y += 58
    for line in lines:
        draw.text((padding + 4, cursor_y), line, fill=(51, 65, 85), font=BODY_FONT)
        cursor_y += 38
    if image_panel is not None:
        cursor_y += 6
        card.paste(image_panel, (padding + 4, cursor_y))
    return card


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line((start, end), fill=color, width=8)
    head = 16
    ex, ey = end
    draw.polygon(
        [(ex, ey), (ex - head, ey - head), (ex + head, ey - head)],
        fill=color,
    )


def render_step_flow(
    blocks: list[ThoughtBlock],
    reasoning_images: dict[str, Image.Image],
    mapped_boxes: list[tuple[int, int, int, int]],
    panel_width: int,
) -> Image.Image:
    gap = 28
    arrow_gap = 54
    cards: list[Image.Image] = []
    for step_number, block in enumerate(blocks, start=1):
        reasoning_image = None
        if block.kind == "j" and block.image_key is not None:
            reasoning_image = reasoning_images[block.image_key]
        cards.append(
            render_step_card(
                step_number=step_number,
                block=block,
                card_width=panel_width,
                reasoning_image=reasoning_image,
                mapped_boxes=mapped_boxes if reasoning_image is not None else None,
            )
        )
    height = sum(card.height for card in cards) + gap * (len(cards) - 1) + arrow_gap * (len(cards) - 1)
    flow = Image.new("RGB", (panel_width, height), (246, 248, 252))
    draw = ImageDraw.Draw(flow)
    cursor_y = 0
    centers: list[tuple[int, int]] = []
    for card in cards:
        flow.paste(card, (0, cursor_y))
        centers.append((panel_width // 2, cursor_y + card.height))
        cursor_y += card.height
        if card is not cards[-1]:
            draw_arrow(
                draw,
                (panel_width // 2, cursor_y + 6),
                (panel_width // 2, cursor_y + arrow_gap - 6),
                color=(148, 163, 184),
            )
            cursor_y += arrow_gap + gap
    return flow


def build_summary_board(
    problem_image: Image.Image,
    question: str,
    blocks: list[ThoughtBlock],
    reasoning_images: dict[str, Image.Image],
    final_answer: str,
    board_width: int,
    show_final_answer: bool,
) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    margin = 42
    gutter = 34
    left_width = int(board_width * 0.49)
    right_width = board_width - margin * 2 - gutter - left_width
    primary_box = find_primary_problem_panel(problem_image)

    mapped_boxes: list[tuple[int, int, int, int]] = []
    first_joint = next((block for block in blocks if block.kind == "j" and block.image_key in reasoning_images), None)
    missing_boxes: list[tuple[int, int, int, int]] = []
    if first_joint is not None:
        if primary_box is not None:
            missing_boxes = detect_missing_regions_from_reference(
                problem_image=problem_image,
                primary_box=primary_box,
                reference_image=reasoning_images[first_joint.image_key],
            )
        if not missing_boxes:
            missing_boxes = detect_missing_regions(problem_image)
        mapped_boxes = map_boxes_to_reasoning(missing_boxes, primary_box, reasoning_images[first_joint.image_key].size)
    else:
        missing_boxes = detect_missing_regions(problem_image)

    left_panel = render_problem_panel(problem_image, question, missing_boxes, left_width)
    right_flow = render_step_flow(blocks, reasoning_images, mapped_boxes, right_width)

    header_h = 132
    body_h = max(left_panel.height, right_flow.height)
    footer_h = 96 if show_final_answer else 42
    board_h = margin + header_h + body_h + footer_h + margin
    board = Image.new("RGB", (board_width, board_h), (246, 248, 252))
    draw = ImageDraw.Draw(board)

    draw.text((margin, margin), "Stage2 Summary Board", fill=(15, 23, 42), font=DISPLAY_FONT)
    subtitle = "Single-canvas chain compression with arrows and local focus highlights"
    draw.text((margin, margin + 62), subtitle, fill=(71, 85, 105), font=SUBTITLE_FONT)

    body_y = margin + header_h
    board.paste(left_panel, (margin, body_y))
    board.paste(right_flow, (margin + left_width + gutter, body_y))

    if show_final_answer:
        footer_y = board_h - margin - 58
        answer_text = f"FINAL ANSWER: {final_answer}"
        badge_w = draw.textbbox((0, 0), answer_text, font=SECTION_FONT)[2] + 54
        badge_x = board_width - margin - badge_w
        draw.rounded_rectangle((badge_x, footer_y, badge_x + badge_w, footer_y + 52), radius=22, fill=(15, 23, 42))
        draw.text((badge_x + 26, footer_y + 10), answer_text, fill=(255, 255, 255), font=SECTION_FONT)

    return board, missing_boxes


def build_record(
    sample_id: int,
    source_subset: str,
    source_sample_index: int,
    question: str,
    final_answer: str,
    problem_image_rel: str,
    board_image_rel: str,
    thought_pattern: list[str],
) -> dict:
    return {
        "metadata": {
            "dataset_name": "Zebra_CoT_summary_board_demo",
            "sample_id": sample_id,
            "source_subset": source_subset,
            "source_sample_index": source_sample_index,
            "thought_pattern": thought_pattern,
            "latent_canvas_count": 1,
            "latent_canvas_paths": [board_image_rel],
            "problem_image_path": problem_image_rel,
            "assistant_text_source": "Final Answer",
            "render_strategy": "single_summary_board_arrow_highlight_v1",
        },
        "data": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": problem_image_rel},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image", "image": board_image_rel},
                    {"type": "text", "text": f"Therefore, the final asnwer is \\boxed{{{final_answer}}}."},
                ],
            },
        ],
    }


def main() -> None:
    args = parse_args()
    record = load_stage2_record(args.stage2_jsonl, args.sample_id)
    metadata = record["metadata"]
    source_subset = metadata["source_subset"]
    source_sample_index = metadata["source_sample_index"]
    dataset_dir = DATASET_DIRS.get(source_subset)
    if dataset_dir is None:
        raise KeyError(f"Unsupported source subset: {source_subset}")

    source_row = fetch_source_row(dataset_dir, source_sample_index)
    question = normalize_question(source_row["Question"] or "")
    final_answer = extract_final_answer(source_row["Final Answer"] or "")
    blocks = thought_sequence(source_row["Text Reasoning Trace"] or "")

    reasoning_images: dict[str, Image.Image] = {}
    for block in blocks:
        if block.kind == "j" and block.image_key is not None:
            struct = source_row.get(block.image_key)
            if struct is None or struct.get("bytes") is None:
                raise ValueError(f"Missing bytes for {block.image_key} in source sample {source_sample_index}.")
            reasoning_images[block.image_key] = Image.open(io.BytesIO(struct["bytes"])).convert("RGB")

    problem_bytes = source_row["problem_image_1"]["bytes"]
    problem_image = Image.open(io.BytesIO(problem_bytes)).convert("RGB")
    board, missing_boxes = build_summary_board(
        problem_image=problem_image,
        question=question,
        blocks=blocks,
        reasoning_images=reasoning_images,
        final_answer=final_answer,
        board_width=args.board_width,
        show_final_answer=args.show_final_answer,
    )

    sample_dir = args.output_root / f"sample_{args.sample_id:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    problem_path = sample_dir / f"problem{detect_image_extension(problem_bytes)}"
    board_path = sample_dir / "assistant_summary_board.png"
    meta_path = sample_dir / "example.json"
    analysis_path = sample_dir / "render_info.json"

    problem_path.write_bytes(problem_bytes)
    board.save(board_path, compress_level=1)

    example_record = build_record(
        sample_id=args.sample_id,
        source_subset=source_subset,
        source_sample_index=source_sample_index,
        question=question,
        final_answer=final_answer,
        problem_image_rel=str(problem_path),
        board_image_rel=str(board_path),
        thought_pattern=[block.kind for block in blocks],
    )
    meta_path.write_text(json.dumps(example_record, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis_path.write_text(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "source_subset": source_subset,
                "source_sample_index": source_sample_index,
                "thought_pattern": [block.kind for block in blocks],
                "missing_region_count": len(missing_boxes),
                "missing_regions": missing_boxes,
                "output_board": str(board_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved summary board to {board_path}")
    print(f"Saved example json to {meta_path}")


if __name__ == "__main__":
    main()
