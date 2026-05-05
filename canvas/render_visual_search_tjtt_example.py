from __future__ import annotations

from PIL import Image

from render_stage2_summary_board_example import (
    ThoughtBlock,
    extract_final_answer,
    normalize_question,
    render_step_card,
    thought_sequence,
)


TARGET_PATTERN = ("t", "j", "t", "t")


def _step_number(block: ThoughtBlock) -> int:
    return int(block.index_label) if str(block.index_label).isdigit() else 0


def make_box(block: ThoughtBlock, canvas_width: int, reasoning_image: Image.Image | None = None) -> Image.Image:
    return render_step_card(
        step_number=_step_number(block),
        block=block,
        card_width=canvas_width,
        reasoning_image=reasoning_image,
    )


def stack_boxes(boxes: list[Image.Image], canvas_width: int, gap: int = 24, padding: int = 24) -> Image.Image:
    height = padding * 2 + sum(box.height for box in boxes) + gap * max(0, len(boxes) - 1)
    canvas = Image.new("RGB", (canvas_width, height), (248, 250, 252))
    cursor_y = padding
    for box in boxes:
        x = max(0, (canvas_width - box.width) // 2)
        canvas.paste(box, (x, cursor_y))
        cursor_y += box.height + gap
    return canvas
