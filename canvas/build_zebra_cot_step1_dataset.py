import argparse
import io
import json
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from render_compact_tj_example import (
    DATASET_ALIAS,
    PATTERN_MAP,
    build_compact_canvas,
    compact_highlight_image,
)
from render_stage2_summary_board_example import (
    DATASET_DIRS,
    dataset_columns,
    detect_image_extension,
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
OUTPUT_ROOT = Path(os.environ.get("UNIVLR_STEP1_OUTPUT_ROOT", str(DATA_ROOT / "Zebra_CoT_step1")))
TARGET_PATTERNS = {PATTERN_MAP["tjt"], PATTERN_MAP["tjtt"]}
DATASET_ORDER = [
    "2D Visual Reasoning - Visual Jigsaw",
    "2D Visual Reasoning - Visual Search",
    "Visual Logic & Strategic Games - Maze",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Monet-format Zebra_CoT_step1 dataset from Visual Jigsaw, "
            "Visual Search, and Maze samples with (t,j,t) or (t,j,t,t) patterns."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory containing the exported images/ and train.json.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory used to store exported problem and assistant images.",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="Output train.json path.",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=1536,
        help="Width for assistant compact canvases.",
    )
    parser.add_argument(
        "--max-body-height",
        type=int,
        default=1180,
        help="Maximum body height for the joint panel before layout balancing.",
    )
    parser.add_argument(
        "--assistant-quality",
        type=int,
        default=90,
        help="JPEG quality for assistant canvases. Lower is smaller/faster.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet batch size while scanning source datasets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 4))),
        help="Worker threads used for rendering and writing images.",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Maximum queued render tasks before flushing ordered results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on exported samples for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing images/train.json before exporting.",
    )
    return parser.parse_args()


def prepare_output_paths(output_root: Path, images_dir: Path, train_json: Path, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if train_json.exists():
            train_json.unlink()
    else:
        if images_dir.exists() and any(images_dir.iterdir()):
            raise FileExistsError(f"{images_dir} already exists and is not empty. Use --overwrite.")
        if train_json.exists():
            raise FileExistsError(f"{train_json} already exists. Use --overwrite.")
    images_dir.mkdir(parents=True, exist_ok=True)


def relative_image_path(output_root: Path, image_path: Path) -> str:
    return image_path.relative_to(output_root.parent).as_posix()


def build_question_text(question: str) -> str:
    question = question.strip()
    suffix = "Put your final answer within \\boxed{}."
    if question.endswith(suffix):
        return question
    if question.endswith("?"):
        return f"{question}\n{suffix}"
    return f"{question}\n\n{suffix}"


def build_record(
    sample_id: int,
    source_subset: str,
    source_sample_index: int,
    thought_pattern: list[str],
    user_image_path: str,
    assistant_image_path: str,
    question: str,
    final_answer: str,
) -> dict:
    return {
        "metadata": {
            "dataset_name": "Zebra_CoT_step1",
            "sample_id": sample_id,
            "source_subset": source_subset,
            "source_sample_index": source_sample_index,
            "thought_pattern": thought_pattern,
        },
        "data": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": user_image_path},
                    {"type": "text", "text": build_question_text(question)},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image", "image": assistant_image_path},
                    {"type": "text", "text": f"Therefore, the final answer is \\boxed{{{final_answer}}}."},
                ],
            },
        ],
    }


def render_sample_canvas(
    dataset_name: str,
    row: dict,
    canvas_width: int,
    max_body_height: int,
) -> Image.Image:
    blocks = row["_blocks"]
    joint_block = next(block for block in blocks if block.kind == "j")
    joint_struct = row[joint_block.image_key]
    joint_image = Image.open(io.BytesIO(joint_struct["bytes"])).convert("RGB")

    problem_probe = Image.open(io.BytesIO(row["problem_image_1"]["bytes"]))
    problem_size = problem_probe.size
    should_skip_detection = dataset_name == "2D Visual Reasoning - Visual Search" and joint_image.size != problem_size
    if should_skip_detection:
        problem_probe.close()
        mapped_boxes = []
    else:
        problem_image = problem_probe.convert("RGB")
        primary_box = find_primary_problem_panel(problem_image)
        if primary_box is not None:
            missing_boxes = detect_missing_regions_from_reference(problem_image, primary_box, joint_image)
        else:
            missing_boxes = []
        if not missing_boxes:
            missing_boxes = detect_missing_regions(problem_image)
        mapped_boxes = map_boxes_to_reasoning(missing_boxes, primary_box, joint_image.size) if primary_box else []

    highlighted_joint = compact_highlight_image(joint_image, mapped_boxes, outline=(16, 185, 129), fill_alpha=36)
    joint_block_tuple = (joint_block.index_label, joint_block.body)
    text_blocks = [
        (block.index_label, block.body, order_index)
        for order_index, block in enumerate(blocks)
        if block.kind == "t"
    ]
    sequence = [(order_index, block.kind) for order_index, block in enumerate(blocks)]
    return build_compact_canvas(
        highlighted_joint=highlighted_joint,
        joint_block=joint_block_tuple,
        text_blocks=text_blocks,
        thought_sequence=sequence,
        canvas_width=canvas_width,
        max_body_height=max_body_height,
    )


def render_and_save_sample(
    task: dict,
    output_root: Path,
    images_dir: Path,
    canvas_width: int,
    max_body_height: int,
    assistant_quality: int,
) -> dict:
    sample_id = task["sample_id"]
    problem_ext = detect_image_extension(task["problem_bytes"])
    user_image_path = images_dir / f"{sample_id}_0{problem_ext}"
    assistant_image_path = images_dir / f"{sample_id}_1.jpg"

    user_image_path.write_bytes(task["problem_bytes"])

    canvas = render_sample_canvas(
        dataset_name=task["source_subset"],
        row=task["row"],
        canvas_width=canvas_width,
        max_body_height=max_body_height,
    )
    canvas.convert("RGB").save(assistant_image_path, quality=assistant_quality, subsampling=0)

    return build_record(
        sample_id=sample_id,
        source_subset=task["source_subset"],
        source_sample_index=task["source_sample_index"],
        thought_pattern=task["thought_pattern"],
        user_image_path=relative_image_path(output_root, user_image_path),
        assistant_image_path=relative_image_path(output_root, assistant_image_path),
        question=task["question"],
        final_answer=task["final_answer"],
    )


def write_record(train_handle, record: dict, first_record: bool) -> bool:
    if not first_record:
        train_handle.write(",")
    json.dump(record, train_handle, ensure_ascii=False, separators=(",", ":"))
    return False


def flush_future_queue(
    pending_futures: set[Future],
    train_handle,
    first_record: bool,
    wait_for_all: bool = False,
) -> tuple[int, bool]:
    written = 0
    while pending_futures:
        if wait_for_all:
            done, pending_futures = wait(pending_futures)
        else:
            done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
        for future in done:
            record = future.result()
            first_record = write_record(train_handle, record, first_record)
            written += 1
        if not wait_for_all:
            break
    if isinstance(pending_futures, set):
        return written, first_record
    return written, first_record


def flush_single_completed_future(pending_futures: set[Future], train_handle, first_record: bool) -> tuple[int, bool]:
    if not pending_futures:
        return 0, first_record
    done, pending = wait(pending_futures, return_when=FIRST_COMPLETED)
    pending_futures.clear()
    pending_futures.update(pending)
    written = 0
    for future in done:
        record = future.result()
        first_record = write_record(train_handle, record, first_record)
        written += 1
    return written, first_record


def main() -> None:
    args = parse_args()
    if args.images_dir is None:
        args.images_dir = args.output_root / "images"
    if args.train_json is None:
        args.train_json = args.output_root / "train.json"
    if args.max_pending is None:
        args.max_pending = max(8, args.workers * 4)

    prepare_output_paths(args.output_root, args.images_dir, args.train_json, args.overwrite)

    total_exported = 0
    per_dataset_counts = {name: 0 for name in DATASET_ORDER}
    executor = None if args.workers <= 1 else ThreadPoolExecutor(max_workers=args.workers)
    pending_futures: set[Future] = set()
    first_record = True

    try:
        with args.train_json.open("w", encoding="utf-8") as train_handle:
            train_handle.write("[")
            for dataset_name in DATASET_ORDER:
                dataset_dir = DATASET_DIRS[dataset_name]
                columns = dataset_columns(dataset_dir)
                source_sample_index = 0
                for parquet_path in sorted(dataset_dir.glob("*.parquet")):
                    parquet = pq.ParquetFile(parquet_path)
                    for batch in parquet.iter_batches(columns=columns, batch_size=args.batch_size):
                        for row_index in range(batch.num_rows):
                            row = row_from_batch(batch, row_index, columns)
                            blocks = thought_sequence(row["Text Reasoning Trace"] or "")
                            pattern = tuple(block.kind for block in blocks)
                            if pattern in TARGET_PATTERNS:
                                row["_blocks"] = blocks
                                task = {
                                    "sample_id": total_exported,
                                    "source_subset": dataset_name,
                                    "source_sample_index": source_sample_index,
                                    "thought_pattern": list(pattern),
                                    "question": normalize_question(row["Question"] or ""),
                                    "final_answer": extract_final_answer(row["Final Answer"] or ""),
                                    "problem_bytes": row["problem_image_1"]["bytes"],
                                    "row": row,
                                }
                                if executor is None:
                                    record = render_and_save_sample(
                                        task=task,
                                        output_root=args.output_root,
                                        images_dir=args.images_dir,
                                        canvas_width=args.canvas_width,
                                        max_body_height=args.max_body_height,
                                        assistant_quality=args.assistant_quality,
                                    )
                                    first_record = write_record(train_handle, record, first_record)
                                else:
                                    pending_futures.add(
                                        executor.submit(
                                            render_and_save_sample,
                                            task,
                                            args.output_root,
                                            args.images_dir,
                                            args.canvas_width,
                                            args.max_body_height,
                                            args.assistant_quality,
                                        )
                                    )
                                    if len(pending_futures) >= args.max_pending:
                                        _, first_record = flush_single_completed_future(
                                            pending_futures, train_handle, first_record
                                        )
                                total_exported += 1
                                per_dataset_counts[dataset_name] += 1

                                if total_exported % 500 == 0:
                                    print(f"exported {total_exported} samples")
                                if args.limit is not None and total_exported >= args.limit:
                                    _, first_record = flush_future_queue(
                                        pending_futures, train_handle, first_record, wait_for_all=True
                                    )
                                    train_handle.write("]\n")
                                    print("done")
                                    print(json.dumps(per_dataset_counts, ensure_ascii=False, indent=2))
                                    return
                            source_sample_index += 1
            _, first_record = flush_future_queue(
                pending_futures, train_handle, first_record, wait_for_all=True
            )
            train_handle.write("]\n")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(f"exported total: {total_exported}")
    print(json.dumps(per_dataset_counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
