#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_compact_tj_example import compact_highlight_image
from render_stage2_summary_board_example import (
    DATASET_DIRS,
    detect_missing_regions,
    detect_missing_regions_from_reference,
    extract_final_answer,
    find_primary_problem_panel,
    map_boxes_to_reasoning,
    normalize_question,
    row_from_batch,
    thought_sequence,
)


DATA_ROOT = Path(os.environ.get("UNIVLR_DATA_ROOT", "data"))
DEFAULT_INPUT_MANIFEST = Path(
    os.environ.get(
        "UNIVLR_STEP1_MANIFEST",
        str(DATA_ROOT / "Zebra_CoT_step1/qwen2_5_vl_latent_targets_24token_2dpool/train_offline_k24.json"),
    )
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("UNIVLR_VERTICAL_ABLATION_OUTPUT_ROOT", str(DATA_ROOT / "Zebra_CoT_step1_vertical_ablation"))
)
DEFAULT_DATASET_NAME = "Zebra_CoT_step1_vertical_ablation"
TARGET_PATTERNS = {
    "tjt": ("t", "j", "t"),
    "tjtt": ("t", "j", "t", "t"),
}

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True)
class RenderOptions:
    input_manifest: Path
    output_root: Path
    images_dir: Path
    image_format: str
    jpeg_quality: int
    png_compress_level: int
    min_canvas_width: int
    max_joint_width: int
    outer_padding: int
    gap: int
    num_latents: int | None
    dataset_name: str
    disable_highlight: bool
    resume: bool


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE_FONT = get_font(20, bold=True)
CHIP_FONT = get_font(14, bold=True)
BODY_FONT = get_font(19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the filtered Zebra_CoT_step1 tjt/tjtt manifest into one vertical "
            "long assistant target image per sample. The output train.json is "
            "UniVLR-compatible: prompt messages + one image step + answer."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
        help="Filtered train_offline_k*.json/jsonl manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output dataset root. Defaults to a sibling under Monet-SFT-125K.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory for rendered assistant images. Default: <output-root>/images.",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="Output UniVLR online manifest. Default: <output-root>/train.json.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="metadata.dataset_name for the rendered ablation dataset.",
    )
    parser.add_argument(
        "--patterns",
        default="tjt,tjtt",
        help="Comma-separated target patterns to render. Supported: tjt,tjtt.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start manifest index, inclusive.")
    parser.add_argument("--end", type=int, default=None, help="End manifest index, exclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after start/end/pattern filtering.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(1, (os.cpu_count() or 4))),
        help="Thread workers for image rendering and writing.",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Maximum queued render tasks. Default: max(16, workers * 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet batch size while scanning source datasets.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "jpeg", "png"),
        default="jpg",
        help="Rendered assistant image format. jpg is faster/smaller; png is lossless.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument(
        "--min-canvas-width",
        type=int,
        default=420,
        help="Lower bound for text legibility. Width still follows the joint image above this floor.",
    )
    parser.add_argument(
        "--max-joint-width",
        type=int,
        default=0,
        help="Optional cap for rendered joint image width. 0 keeps the source joint width.",
    )
    parser.add_argument("--outer-padding", type=int, default=24)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument(
        "--num-latents",
        type=int,
        default=None,
        help="Override num_latents in output steps. Default: reuse input step num_latents.",
    )
    parser.add_argument(
        "--disable-highlight",
        action="store_true",
        help="Do not run Canvas_V0 missing-region highlight on the joint image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing images/train.json before rendering.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow an existing images directory and reuse already-rendered canvas files.",
    )
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def parse_target_patterns(raw: str) -> set[tuple[str, ...]]:
    patterns: set[tuple[str, ...]] = set()
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in TARGET_PATTERNS:
            raise ValueError(f"Unsupported pattern {key!r}; supported: {sorted(TARGET_PATTERNS)}")
        patterns.add(TARGET_PATTERNS[key])
    if not patterns:
        raise ValueError("No target patterns were selected.")
    return patterns


def prepare_output_paths(output_root: Path, images_dir: Path, train_json: Path, overwrite: bool, resume: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if train_json.exists():
            train_json.unlink()
    elif not resume:
        if images_dir.exists() and any(images_dir.iterdir()):
            raise FileExistsError(f"{images_dir} already exists and is not empty. Use --overwrite or --resume.")
        if train_json.exists():
            raise FileExistsError(f"{train_json} already exists. Use --overwrite or --resume.")
    images_dir.mkdir(parents=True, exist_ok=True)


def media_path_for_manifest(output_root: Path, path: Path) -> str:
    try:
        return path.relative_to(output_root.parent).as_posix()
    except ValueError:
        return str(path)


def sort_reasoning_columns(columns: list[str] | set[str]) -> list[str]:
    def order_key(name: str) -> int:
        suffix = name.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 10**9

    return sorted((c for c in columns if c.startswith("reasoning_image_")), key=order_key)


def source_schema_names(dataset_dir: Path) -> list[str]:
    first_file = sorted(dataset_dir.glob("*.parquet"))[0]
    return pq.ParquetFile(first_file).schema_arrow.names


def source_columns(dataset_dir: Path, reasoning_columns: list[str] | set[str] | None = None) -> list[str]:
    schema_names = set(source_schema_names(dataset_dir))
    if reasoning_columns is None:
        reasoning_columns = sort_reasoning_columns(schema_names)
    else:
        missing = sorted(set(reasoning_columns) - schema_names)
        if missing:
            raise ValueError(f"Missing reasoning columns in {dataset_dir}: {missing}")
        reasoning_columns = sort_reasoning_columns(reasoning_columns)
    return ["Final Answer", "problem_image_1", *reasoning_columns]


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def split_long_word(draw: ImageDraw.ImageDraw, word: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if text_size(draw, word, font)[0] <= max_width:
        return [word]
    pieces: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if current and text_size(draw, candidate, font)[0] > max_width:
            pieces.append(current)
            current = char
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [word]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    paragraphs = (text or " ").splitlines() or [""]
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for raw_word in words:
            for word in split_long_word(draw, raw_word, font, max_width):
                candidate = word if not current else f"{current} {word}"
                if not current or text_size(draw, candidate, font)[0] <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
        if current:
            lines.append(current)
    return lines or [""]


def resize_to_width(image: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or image.width <= max_width:
        return image.copy()
    ratio = max_width / image.width
    new_size = (max(1, int(round(image.width * ratio))), max(1, int(round(image.height * ratio))))
    return image.resize(new_size, Image.LANCZOS)


def draw_chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
    ink: tuple[int, int, int],
) -> int:
    x, y = xy
    w, _ = text_size(draw, text, CHIP_FONT)
    rect = (x, y, x + w + 18, y + 24)
    draw.rounded_rectangle(rect, radius=9, fill=fill)
    draw.text((x + 9, y + 4), text, fill=ink, font=CHIP_FONT)
    return rect[2]


def make_block_card(block: Any, image: Image.Image | None, canvas_width: int, opts: RenderOptions) -> Image.Image:
    strip_w = 8
    card_pad_x = 18
    card_pad_y = 16
    line_gap = 6
    image_gap = 14
    radius = 14

    probe = Image.new("RGB", (canvas_width, 10), "white")
    draw = ImageDraw.Draw(probe)
    card_w = canvas_width - opts.outer_padding * 2
    inner_w = max(32, card_w - card_pad_x * 2 - strip_w)
    lines = wrap_text(draw, block.body, BODY_FONT, inner_w)
    line_h = text_size(draw, "Ag", BODY_FONT)[1]
    body_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap
    image_h = image.height + image_gap if image is not None else 0
    card_h = card_pad_y + 28 + 34 + body_h + image_h + card_pad_y

    card = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    d = ImageDraw.Draw(card)
    is_joint = block.kind == "j"
    accent = (16, 185, 129) if is_joint else (59, 130, 246)
    outline = (187, 247, 208) if is_joint else (191, 219, 254)
    chip_fill = (220, 252, 231) if is_joint else (219, 234, 254)
    chip_ink = (22, 101, 52) if is_joint else (30, 64, 175)
    kind_text = "JOINT" if is_joint else "TEXT"

    d.rounded_rectangle((0, 0, card_w - 1, card_h - 1), radius=radius, fill=(255, 255, 255), outline=outline, width=2)
    d.rounded_rectangle((0, 0, strip_w + radius, card_h - 1), radius=radius, fill=accent)
    d.rectangle((strip_w, 0, strip_w + radius, card_h - 1), fill=accent)

    x = strip_w + card_pad_x
    y = card_pad_y
    d.text((x, y), f"THOUGHT {block.index_label}", fill=(15, 23, 42), font=TITLE_FONT)
    y += 32
    draw_chip(d, (x, y), kind_text, chip_fill, chip_ink)
    y += 36
    for line in lines:
        d.text((x, y), line, fill=(55, 65, 81), font=BODY_FONT)
        y += line_h + line_gap

    if image is not None:
        y += image_gap - line_gap
        img_x = (card_w - image.width) // 2
        card.paste(image, (img_x, y))
    return card


def make_vertical_canvas(blocks: list[Any], joint_image: Image.Image, opts: RenderOptions) -> Image.Image:
    canvas_w = max(opts.min_canvas_width, joint_image.width + opts.outer_padding * 2)
    cards = [make_block_card(block, joint_image if block.kind == "j" else None, canvas_w, opts) for block in blocks]
    canvas_h = opts.outer_padding + sum(card.height for card in cards) + opts.gap * (len(cards) - 1) + opts.outer_padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), (246, 248, 252))
    draw = ImageDraw.Draw(canvas)
    y = opts.outer_padding
    for idx, card in enumerate(cards):
        canvas.paste(card, (opts.outer_padding, y))
        if idx < len(cards) - 1:
            cx = canvas_w // 2
            y2 = y + card.height + opts.gap
            draw.line((cx, y + card.height + 4, cx, y2 - 8), fill=(148, 163, 184), width=3)
            draw.polygon([(cx, y2 - 4), (cx - 7, y2 - 12), (cx + 7, y2 - 12)], fill=(148, 163, 184))
        y += card.height + opts.gap
    return canvas


def highlighted_joint_image(
    dataset_name: str,
    problem_bytes: bytes,
    joint_image: Image.Image,
    opts: RenderOptions,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    if opts.disable_highlight:
        return joint_image.copy(), [], []

    problem_probe = Image.open(io.BytesIO(problem_bytes))
    should_skip_detection = (
        dataset_name == "2D Visual Reasoning - Visual Search" and joint_image.size != problem_probe.size
    )
    if should_skip_detection:
        return joint_image.copy(), [], []

    problem_image = problem_probe.convert("RGB")
    primary_box = find_primary_problem_panel(problem_image)
    if primary_box is not None:
        missing_boxes = detect_missing_regions_from_reference(problem_image, primary_box, joint_image)
    else:
        missing_boxes = []
    if not missing_boxes:
        missing_boxes = detect_missing_regions(problem_image)
    mapped_boxes = map_boxes_to_reasoning(missing_boxes, primary_box, joint_image.size) if primary_box else []
    highlighted = compact_highlight_image(joint_image, mapped_boxes, outline=(16, 185, 129), fill_alpha=36)
    return highlighted, missing_boxes, mapped_boxes


def save_canvas(canvas: Image.Image, path: Path, opts: RenderOptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if opts.image_format in {"jpg", "jpeg"}:
        canvas.convert("RGB").save(path, quality=opts.jpeg_quality, subsampling=0)
    else:
        canvas.save(path, compress_level=opts.png_compress_level)


def prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = copy.deepcopy(record.get("messages") or record.get("data") or [])
    return [message for message in messages if message.get("role") != "assistant"]


def build_answer(record: dict[str, Any], source_row: dict[str, Any]) -> str:
    answer = record.get("answer")
    if answer:
        return str(answer)
    final_answer = extract_final_answer(source_row.get("Final Answer") or "")
    return f"Therefore, the final answer is \\boxed{{{final_answer}}}."


def output_step(record: dict[str, Any], image_rel: str, opts: RenderOptions) -> dict[str, Any]:
    source_steps = record.get("steps") or record.get("latent_steps") or record.get("cot_steps") or []
    source_step = source_steps[0] if source_steps else {}
    num_latents = opts.num_latents
    if num_latents is None:
        num_latents = source_step.get("num_latents") or source_step.get("num_image_latents")
    step: dict[str, Any] = {"kind": "image", "image": image_rel}
    if num_latents is not None:
        step["num_latents"] = int(num_latents)
    return step


def render_one(
    input_index: int,
    record: dict[str, Any],
    source_row: dict[str, Any],
    dataset_name: str,
    blocks: list[Any],
    opts: RenderOptions,
) -> tuple[int, dict[str, Any]]:
    suffix = ".jpg" if opts.image_format in {"jpg", "jpeg"} else ".png"
    image_path = opts.images_dir / f"sample_{input_index:06d}_assistant_vertical{suffix}"
    image_rel = media_path_for_manifest(opts.output_root, image_path)

    joint_block = next(block for block in blocks if block.kind == "j")
    joint_struct = source_row.get(joint_block.image_key)
    if not joint_struct or "bytes" not in joint_struct:
        raise ValueError(f"Missing bytes for {joint_block.image_key} at manifest index {input_index}.")

    joint_image = Image.open(io.BytesIO(joint_struct["bytes"])).convert("RGB")
    original_joint_size = joint_image.size

    if opts.resume and image_path.exists():
        canvas_size = Image.open(image_path).size
        rendered_joint_size = None
        missing_boxes: list[tuple[int, int, int, int]] = []
        mapped_boxes: list[tuple[int, int, int, int]] = []
    else:
        highlighted, missing_boxes, mapped_boxes = highlighted_joint_image(
            dataset_name=dataset_name,
            problem_bytes=source_row["problem_image_1"]["bytes"],
            joint_image=joint_image,
            opts=opts,
        )
        rendered_joint = resize_to_width(highlighted, opts.max_joint_width)
        rendered_joint_size = rendered_joint.size
        canvas = make_vertical_canvas(blocks, rendered_joint, opts)
        canvas_size = canvas.size
        save_canvas(canvas, image_path, opts)

    metadata = copy.deepcopy(record.get("metadata") or {})
    source_dataset_name = metadata.get("dataset_name")
    source_steps = record.get("steps") or record.get("latent_steps") or record.get("cot_steps") or []
    source_target_path = None
    if source_steps:
        source_target_path = (
            source_steps[0].get("target_path")
            or source_steps[0].get("image_target_path")
            or source_steps[0].get("text_target_path")
            or source_steps[0].get("latent_target_path")
        )
    metadata.update(
        {
            "dataset_name": opts.dataset_name,
            "source_dataset_name": source_dataset_name,
            "source_manifest": str(opts.input_manifest),
            "source_manifest_index": input_index,
            "source_subset": dataset_name,
            "source_sample_index": metadata.get("source_sample_index"),
            "thought_pattern": [block.kind for block in blocks],
            "latent_canvas_count": 1,
            "latent_canvas_paths": [image_rel],
            "render_strategy": "vertical_tjt_tjtt_width_from_joint_image_v1",
            "joint_image_size": list(original_joint_size),
            "rendered_joint_size": list(rendered_joint_size) if rendered_joint_size else None,
            "canvas_size": list(canvas_size),
            "width_rule": (
                "canvas_width = max(min_canvas_width, rendered_joint_image_width + 2 * outer_padding)"
            ),
            "min_canvas_width": opts.min_canvas_width,
            "max_joint_width": opts.max_joint_width,
            "missing_region_count": len(missing_boxes),
            "mapped_region_count": len(mapped_boxes),
        }
    )
    if source_target_path:
        metadata["source_offline_target_path"] = source_target_path

    output_record = {
        "messages": prompt_messages(record),
        "steps": [output_step(record, image_rel, opts)],
        "answer": build_answer(record, source_row),
        "metadata": metadata,
    }
    return input_index, output_record


def flush_pending(
    pending: set[Future],
    output_records: dict[int, dict[str, Any]],
    wait_for_all: bool = False,
) -> int:
    if not pending:
        return 0
    if wait_for_all:
        done, rest = wait(pending)
    else:
        done, rest = wait(pending, return_when=FIRST_COMPLETED)
    pending.clear()
    pending.update(rest)
    completed = 0
    for future in done:
        input_index, output_record = future.result()
        output_records[input_index] = output_record
        completed += 1
    return completed


def iter_needed_source_rows(dataset_name: str, source_indices: list[int], columns: list[str], batch_size: int):
    dataset_dir = DATASET_DIRS[dataset_name]
    wanted = sorted(set(source_indices))
    wanted_pos = 0
    current_index = 0
    for parquet_path in sorted(dataset_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            batch_start = current_index
            batch_end = current_index + batch.num_rows
            while wanted_pos < len(wanted) and wanted[wanted_pos] < batch_end:
                source_index = wanted[wanted_pos]
                if source_index >= batch_start:
                    row_index = source_index - batch_start
                    yield source_index, row_from_batch(batch, row_index, columns)
                wanted_pos += 1
            current_index = batch_end
            if wanted_pos >= len(wanted):
                return


def collect_source_specs(
    dataset_name: str,
    source_indices: list[int],
    target_patterns: set[tuple[str, ...]],
    batch_size: int,
) -> tuple[dict[int, list[Any]], set[str]]:
    columns = ["Text Reasoning Trace"]
    specs: dict[int, list[Any]] = {}
    reasoning_columns: set[str] = set()
    for source_index, row in iter_needed_source_rows(dataset_name, source_indices, columns, batch_size):
        blocks = thought_sequence(row["Text Reasoning Trace"] or "")
        source_pattern = tuple(block.kind for block in blocks)
        if source_pattern not in target_patterns:
            raise ValueError(f"Source pattern mismatch for {dataset_name} index {source_index}: {source_pattern}")
        joint_blocks = [block for block in blocks if block.kind == "j"]
        if len(joint_blocks) != 1:
            raise ValueError(f"Expected one joint block for {dataset_name} index {source_index}, got {len(joint_blocks)}")
        if not joint_blocks[0].image_key:
            raise ValueError(f"Joint block has no image key for {dataset_name} index {source_index}")
        reasoning_columns.add(joint_blocks[0].image_key)
        specs[source_index] = blocks
    return specs, reasoning_columns


def write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for idx, record in enumerate(records):
            if idx:
                handle.write(",")
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("]\n")


def write_render_info(info: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_manifest = args.input_manifest.resolve()
    if not input_manifest.is_file():
        raise FileNotFoundError(input_manifest)

    output_root = args.output_root.resolve()
    images_dir = (args.images_dir or output_root / "images").resolve()
    train_json = (args.train_json or output_root / "train.json").resolve()
    if args.max_pending is None:
        args.max_pending = max(16, args.workers * 4)

    prepare_output_paths(output_root, images_dir, train_json, args.overwrite, args.resume)
    target_patterns = parse_target_patterns(args.patterns)
    all_records = load_json_or_jsonl(input_manifest)

    start = max(0, args.start)
    end = len(all_records) if args.end is None else min(args.end, len(all_records))
    selected_indices: list[int] = []
    skipped_by_pattern = 0
    for input_index in range(start, end):
        record = all_records[input_index]
        pattern = tuple(record.get("metadata", {}).get("thought_pattern") or [])
        if pattern not in target_patterns:
            skipped_by_pattern += 1
            continue
        selected_indices.append(input_index)
        if args.limit is not None and len(selected_indices) >= args.limit:
            break

    needed_by_subset: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for input_index in selected_indices:
        metadata = all_records[input_index].get("metadata") or {}
        subset = metadata.get("source_subset")
        source_index = metadata.get("source_sample_index")
        if subset not in DATASET_DIRS:
            raise ValueError(f"Unsupported source subset at manifest index {input_index}: {subset!r}")
        if source_index is None:
            raise ValueError(f"Missing source_sample_index at manifest index {input_index}.")
        needed_by_subset[subset][int(source_index)].append(input_index)

    opts = RenderOptions(
        input_manifest=input_manifest,
        output_root=output_root,
        images_dir=images_dir,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        png_compress_level=args.png_compress_level,
        min_canvas_width=args.min_canvas_width,
        max_joint_width=args.max_joint_width,
        outer_padding=args.outer_padding,
        gap=args.gap,
        num_latents=args.num_latents,
        dataset_name=args.dataset_name,
        disable_highlight=args.disable_highlight,
        resume=args.resume,
    )

    print(
        json.dumps(
            {
                "input_manifest": str(input_manifest),
                "output_root": str(output_root),
                "train_json": str(train_json),
                "selected_samples": len(selected_indices),
                "skipped_by_pattern": skipped_by_pattern,
                "workers": args.workers,
                "max_pending": args.max_pending,
                "image_format": args.image_format,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    output_records: dict[int, dict[str, Any]] = {}
    pending: set[Future] = set()
    processed = 0
    next_progress = 500
    started_at = time.time()
    executor = None if args.workers <= 1 else ThreadPoolExecutor(max_workers=args.workers)

    try:
        for dataset_name, source_map in needed_by_subset.items():
            print(f"[scan] {dataset_name}: {len(source_map)} source rows", flush=True)
            source_specs, reasoning_columns = collect_source_specs(
                dataset_name=dataset_name,
                source_indices=list(source_map.keys()),
                target_patterns=target_patterns,
                batch_size=args.batch_size,
            )
            if len(source_specs) != len(source_map):
                missing = sorted(set(source_map) - set(source_specs))
                raise RuntimeError(
                    f"Only found {len(source_specs)}/{len(source_map)} trace rows for {dataset_name}. "
                    f"Missing examples include: {missing[:10]}"
                )
            columns = source_columns(DATASET_DIRS[dataset_name], reasoning_columns)
            print(
                f"[scan] {dataset_name}: reading image columns {sorted(reasoning_columns)}",
                flush=True,
            )
            found_sources = 0
            for source_index, source_row in iter_needed_source_rows(
                dataset_name=dataset_name,
                source_indices=list(source_map.keys()),
                columns=columns,
                batch_size=args.batch_size,
            ):
                found_sources += 1
                blocks = source_specs[source_index]
                for input_index in source_map[source_index]:
                    record = all_records[input_index]
                    if executor is None:
                        _, output_record = render_one(
                            input_index=input_index,
                            record=record,
                            source_row=source_row,
                            dataset_name=dataset_name,
                            blocks=blocks,
                            opts=opts,
                        )
                        output_records[input_index] = output_record
                        processed += 1
                    else:
                        pending.add(
                            executor.submit(
                                render_one,
                                input_index,
                                record,
                                source_row,
                                dataset_name,
                                blocks,
                                opts,
                            )
                        )
                        if len(pending) >= args.max_pending:
                            processed += flush_pending(pending, output_records)
                    if processed >= next_progress:
                        elapsed = max(1e-6, time.time() - started_at)
                        print(
                            f"[render] {processed}/{len(selected_indices)} done "
                            f"({processed / elapsed:.2f} samples/s)",
                            flush=True,
                        )
                        next_progress += 500

            missing_sources = set(source_map) - {
                int(record.get("metadata", {}).get("source_sample_index"))
                for idx, record in output_records.items()
                if idx in selected_indices
                and record.get("metadata", {}).get("source_subset") == dataset_name
            }
            if found_sources != len(source_map):
                raise RuntimeError(
                    f"Only found {found_sources}/{len(source_map)} source rows for {dataset_name}. "
                    f"Missing examples include: {sorted(missing_sources)[:10]}"
                )

        processed += flush_pending(pending, output_records, wait_for_all=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    missing_outputs = [idx for idx in selected_indices if idx not in output_records]
    if missing_outputs:
        raise RuntimeError(f"Missing rendered records for manifest indices: {missing_outputs[:20]}")

    ordered_records = [output_records[idx] for idx in selected_indices]
    write_manifest(ordered_records, train_json)
    elapsed = max(1e-6, time.time() - started_at)
    info = {
        "input_manifest": str(input_manifest),
        "output_root": str(output_root),
        "images_dir": str(images_dir),
        "train_json": str(train_json),
        "samples": len(ordered_records),
        "source_subsets": {name: sum(len(v) for v in source_map.values()) for name, source_map in needed_by_subset.items()},
        "elapsed_seconds": elapsed,
        "samples_per_second": len(ordered_records) / elapsed,
        "render_strategy": "vertical_tjt_tjtt_width_from_joint_image_v1",
        "image_format": args.image_format,
        "min_canvas_width": args.min_canvas_width,
        "max_joint_width": args.max_joint_width,
        "patterns": ["".join(pattern) for pattern in sorted(target_patterns)],
    }
    write_render_info(info, output_root / "render_info.json")
    print(
        json.dumps(
            {
                "done": True,
                "samples": len(ordered_records),
                "train_json": str(train_json),
                "images_dir": str(images_dir),
                "elapsed_seconds": round(elapsed, 2),
                "samples_per_second": round(len(ordered_records) / elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
