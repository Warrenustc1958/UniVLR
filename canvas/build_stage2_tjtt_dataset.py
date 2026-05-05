import argparse
import io
import json
import os
import shutil
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from PIL import Image

from render_visual_search_tjtt_example import (
    TARGET_PATTERN,
    ThoughtBlock,
    extract_final_answer,
    make_box,
    normalize_question,
    stack_boxes,
    thought_sequence,
)


DATA_ROOT = Path(os.environ.get("UNIVLR_DATA_ROOT", "data"))
VISUAL_JIGSAW_DIR = Path(
    os.environ.get(
        "UNIVLR_VISUAL_JIGSAW_DIR",
        str(DATA_ROOT / "zebra-cot/2D Visual Reasoning - Visual Jigsaw"),
    )
)
VISUAL_SEARCH_DIR = Path(
    os.environ.get(
        "UNIVLR_VISUAL_SEARCH_DIR",
        str(DATA_ROOT / "zebra-cot/2D Visual Reasoning - Visual Search"),
    )
)
OUTPUT_ROOT = Path(
    os.environ.get("UNIVLR_STAGE2_OUTPUT_ROOT", str(DATA_ROOT / "Zebra_CoT_visual_search"))
)
ASSET_DIR = OUTPUT_ROOT / "stage2"
JSONL_PATH = OUTPUT_ROOT / "stage2_train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch export all (t,j,t,t) samples from Visual Jigsaw and Visual Search "
            "into stage2 canvases plus a stage2_train.jsonl file."
        )
    )
    parser.add_argument(
        "--visual-jigsaw-dir",
        type=Path,
        default=VISUAL_JIGSAW_DIR,
        help="Directory containing the Visual Jigsaw parquet shards.",
    )
    parser.add_argument(
        "--visual-search-dir",
        type=Path,
        default=VISUAL_SEARCH_DIR,
        help="Directory containing the Visual Search parquet shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root output directory that will contain stage2/ and stage2_train.jsonl.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=None,
        help="Directory used to store rendered problem/canvas images.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=None,
        help="Output JSONL path for the exported sample structures.",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=1024,
        help="Target width for the rendered vertical canvases.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used when scanning parquet shards.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of exported samples for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing stage2 assets/jsonl before exporting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 4))),
        help="Number of worker threads used for rendering and image writing.",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Maximum number of queued render tasks before flushing results.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=1,
        help="PNG compress level for rendered canvases. Lower is faster.",
    )
    return parser.parse_args()


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


def prepare_output_paths(output_root: Path, asset_dir: Path, jsonl_path: Path, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
        if jsonl_path.exists():
            jsonl_path.unlink()
    else:
        if asset_dir.exists() and any(asset_dir.iterdir()):
            raise FileExistsError(
                f"{asset_dir} already exists and is not empty. Use --overwrite to rebuild it."
            )
        if jsonl_path.exists():
            raise FileExistsError(
                f"{jsonl_path} already exists. Use --overwrite to rebuild it."
            )
    asset_dir.mkdir(parents=True, exist_ok=True)


def row_from_batch(batch, row_index: int, columns: list[str]) -> dict:
    row = {}
    for name in columns:
        row[name] = batch.column(batch.schema.get_field_index(name))[row_index].as_py()
    return row


def render_canvases(blocks: list[ThoughtBlock], row: dict, canvas_width: int) -> tuple[Image.Image, Image.Image]:
    if len(blocks) != 4:
        raise ValueError(f"Expected exactly 4 thoughts for {TARGET_PATTERN}, got {len(blocks)}.")

    joint_block = next(block for block in blocks if block.kind == "j")
    if joint_block.image_key is None:
        raise ValueError("Expected the joint thought to reference a reasoning image.")

    reasoning_struct = row[joint_block.image_key]
    if reasoning_struct is None or reasoning_struct.get("bytes") is None:
        raise ValueError(f"Missing bytes for {joint_block.image_key}.")

    reasoning_image = Image.open(io.BytesIO(reasoning_struct["bytes"])).convert("RGB")

    first_canvas_boxes = [
        make_box(blocks[0], canvas_width, reasoning_image=None),
        make_box(blocks[1], canvas_width, reasoning_image=reasoning_image),
    ]
    second_canvas_boxes = [
        make_box(blocks[2], canvas_width, reasoning_image=None),
        make_box(blocks[3], canvas_width, reasoning_image=None),
    ]
    return (
        stack_boxes(first_canvas_boxes, canvas_width),
        stack_boxes(second_canvas_boxes, canvas_width),
    )


def detect_image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ".webp"
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").lower()
        if image_format in {"jpeg", "jpg"}:
            return ".jpg"
        if image_format == "png":
            return ".png"
        if image_format == "webp":
            return ".webp"
    except Exception:
        pass
    return ".jpg"


def relative_image_path(output_root: Path, asset_path: Path) -> str:
    return asset_path.relative_to(output_root).as_posix()


def build_record(
    stage2_index: int,
    source_subset: str,
    source_sample_index: int,
    question: str,
    final_answer: str,
    problem_image_path: str,
    canvas_paths: list[str],
) -> dict:
    return {
        "metadata": {
            "dataset_name": "Zebra_CoT_visual_search_stage2",
            "sample_id": stage2_index,
            "source_subset": source_subset,
            "source_sample_index": source_sample_index,
            "thought_pattern": list(TARGET_PATTERN),
            "latent_canvas_count": len(canvas_paths),
            "latent_canvas_paths": canvas_paths,
            "problem_image_path": problem_image_path,
            "assistant_text_source": "Final Answer",
            "render_strategy": "vertical_pair_canvas_width_1024",
        },
        "data": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": problem_image_path},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image", "image": canvas_paths[0]},
                    {"type": "image", "image": canvas_paths[1]},
                    {
                        "type": "text",
                        "text": f"Therefore, the final asnwer is \\boxed{{{final_answer}}}.",
                    },
                ],
            },
        ],
    }


def render_and_save_sample(
    task: dict,
    output_root: Path,
    asset_dir: Path,
    canvas_width: int,
    png_compress_level: int,
) -> dict:
    stage2_index = task["stage2_index"]

    problem_extension = detect_image_extension(task["problem_image_bytes"])
    problem_path = asset_dir / f"sample_{stage2_index:06d}_problem{problem_extension}"
    canvas_0_path = asset_dir / f"sample_{stage2_index:06d}_assistant_canvas_0.png"
    canvas_1_path = asset_dir / f"sample_{stage2_index:06d}_assistant_canvas_1.png"

    # Copy the original problem image bytes directly instead of decoding and re-encoding.
    problem_path.write_bytes(task["problem_image_bytes"])

    render_row = {
        "problem_image_1": {"bytes": task["problem_image_bytes"]},
        task["joint_image_key"]: {"bytes": task["reasoning_image_bytes"]},
    }
    canvas_0, canvas_1 = render_canvases(task["blocks"], render_row, canvas_width)
    canvas_0.save(canvas_0_path, compress_level=png_compress_level)
    canvas_1.save(canvas_1_path, compress_level=png_compress_level)

    problem_rel = relative_image_path(output_root, problem_path)
    canvas_rels = [
        relative_image_path(output_root, canvas_0_path),
        relative_image_path(output_root, canvas_1_path),
    ]
    return build_record(
        stage2_index=stage2_index,
        source_subset=task["source_subset"],
        source_sample_index=task["source_sample_index"],
        question=task["question"],
        final_answer=task["final_answer"],
        problem_image_path=problem_rel,
        canvas_paths=canvas_rels,
    )


def flush_future_queue(
    pending_futures: deque[Future],
    jsonl_file,
) -> int:
    written = 0
    while pending_futures:
        future = pending_futures.popleft()
        record = future.result()
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
    return written


def export_dataset(
    subset_name: str,
    dataset_dir: Path,
    output_root: Path,
    asset_dir: Path,
    jsonl_file,
    canvas_width: int,
    batch_size: int,
    start_stage2_index: int,
    limit: int | None,
    executor: ThreadPoolExecutor | None,
    max_pending: int,
    png_compress_level: int,
) -> tuple[int, int]:
    columns = dataset_columns(dataset_dir)
    files = sorted(dataset_dir.glob("*.parquet"))
    exported = 0
    source_sample_index = 0
    pending_futures: deque[Future] = deque()

    for file_path in files:
        parquet = pq.ParquetFile(file_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            for row_index in range(batch.num_rows):
                row = row_from_batch(batch, row_index, columns)
                blocks = thought_sequence(row["Text Reasoning Trace"] or "")
                pattern = tuple(block.kind for block in blocks)

                if pattern == TARGET_PATTERN:
                    stage2_index = start_stage2_index + exported
                    joint_block = next(block for block in blocks if block.kind == "j")
                    if joint_block.image_key is None:
                        raise ValueError(f"{subset_name} row {source_sample_index} has no reasoning image key.")
                    reasoning_struct = row[joint_block.image_key]
                    if reasoning_struct is None or reasoning_struct.get("bytes") is None:
                        raise ValueError(
                            f"{subset_name} row {source_sample_index} is missing bytes for {joint_block.image_key}."
                        )

                    task = {
                        "stage2_index": stage2_index,
                        "source_subset": subset_name,
                        "source_sample_index": source_sample_index,
                        "question": normalize_question(row["Question"]),
                        "final_answer": extract_final_answer(row["Final Answer"]),
                        "blocks": blocks,
                        "problem_image_bytes": row["problem_image_1"]["bytes"],
                        "joint_image_key": joint_block.image_key,
                        "reasoning_image_bytes": reasoning_struct["bytes"],
                    }

                    if executor is None:
                        record = render_and_save_sample(
                            task=task,
                            output_root=output_root,
                            asset_dir=asset_dir,
                            canvas_width=canvas_width,
                            png_compress_level=png_compress_level,
                        )
                        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        pending_futures.append(
                            executor.submit(
                                render_and_save_sample,
                                task,
                                output_root,
                                asset_dir,
                                canvas_width,
                                png_compress_level,
                            )
                        )
                        if len(pending_futures) >= max_pending:
                            future = pending_futures.popleft()
                            record = future.result()
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                    exported += 1

                    if exported % 200 == 0:
                        print(
                            f"[{subset_name}] exported {exported} matched samples "
                            f"(latest stage2 index: {stage2_index})"
                        )

                    if limit is not None and start_stage2_index + exported >= limit:
                        flushed = flush_future_queue(pending_futures, jsonl_file)
                        return exported, source_sample_index + 1

                source_sample_index += 1

    flush_future_queue(pending_futures, jsonl_file)
    return exported, source_sample_index


def main() -> None:
    args = parse_args()
    if args.asset_dir is None:
        args.asset_dir = args.output_root / "stage2"
    if args.jsonl_path is None:
        args.jsonl_path = args.output_root / "stage2_train.jsonl"
    if args.max_pending is None:
        args.max_pending = max(8, args.workers * 4)
    prepare_output_paths(args.output_root, args.asset_dir, args.jsonl_path, args.overwrite)

    datasets = [
        ("2D Visual Reasoning - Visual Jigsaw", args.visual_jigsaw_dir),
        ("2D Visual Reasoning - Visual Search", args.visual_search_dir),
    ]

    total_exported = 0
    per_dataset_counts: list[tuple[str, int]] = []

    executor = None if args.workers <= 1 else ThreadPoolExecutor(max_workers=args.workers)
    try:
        with args.jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for subset_name, dataset_dir in datasets:
                if args.limit is not None and total_exported >= args.limit:
                    break

                exported, scanned_rows = export_dataset(
                    subset_name=subset_name,
                    dataset_dir=dataset_dir,
                    output_root=args.output_root,
                    asset_dir=args.asset_dir,
                    jsonl_file=jsonl_file,
                    canvas_width=args.canvas_width,
                    batch_size=args.batch_size,
                    start_stage2_index=total_exported,
                    limit=args.limit,
                    executor=executor,
                    max_pending=args.max_pending,
                    png_compress_level=args.png_compress_level,
                )
                total_exported += exported
                per_dataset_counts.append((subset_name, exported))
                print(
                    f"Finished {subset_name}: exported {exported} (t,j,t,t) samples "
                    f"after scanning {scanned_rows} source rows."
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(f"stage2 assets written to: {args.asset_dir}")
    print(f"stage2 jsonl written to: {args.jsonl_path}")
    print(f"total exported samples: {total_exported}")
    for subset_name, count in per_dataset_counts:
        print(f"  - {subset_name}: {count}")


if __name__ == "__main__":
    main()
