import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    UniVLR_END_TOKEN,
    UniVLR_LATENT_END_TOKEN,
    UniVLR_START_TOKEN,
    UniVLR_TOKEN,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from src.params import DataArguments

from .data_utils import get_image_info, pad_sequence


SLOT_TYPE_TEXT = 0
SLOT_TYPE_IMAGE = 1
MANIFEST_PATH_KEY = "__univlr_manifest_path"
MANIFEST_DIR_KEY = "__univlr_manifest_dir"


def _load_json_or_jsonl(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _attach_manifest_metadata(data: List[Dict[str, Any]], data_path: str) -> List[Dict[str, Any]]:
    manifest_path = os.path.abspath(data_path)
    manifest_dir = os.path.dirname(manifest_path)
    for sample in data:
        if isinstance(sample, dict):
            sample.setdefault(MANIFEST_PATH_KEY, manifest_path)
            sample.setdefault(MANIFEST_DIR_KEY, manifest_dir)
    return data


def _resolve_path(path: Optional[str], image_folder: Optional[str]) -> Optional[str]:
    if path is None or path == "":
        return None
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
        return path
    if os.path.exists(path):
        return path
    if image_folder:
        joined = os.path.join(image_folder, path)
        if os.path.exists(joined):
            return joined
    return path


def _strip_answer_markup(text: str) -> str:
    text = re.sub(r"<abs_vis_token>.*?</abs_vis_token>", "", text, flags=re.DOTALL)
    text = text.replace("<observation>", "").replace("</observation>", "")
    return text


def _latent_segment(num_tokens: int) -> str:
    return f"{UniVLR_START_TOKEN}{UniVLR_TOKEN * int(num_tokens)}{UniVLR_END_TOKEN}"


def _load_target_payload(path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metadata: Dict[str, Any] = {}
    target = payload
    if isinstance(payload, dict):
        metadata = dict(payload)
        for key in ("latent_targets", "target", "embeds", "embedding", "hidden_states"):
            if key in payload:
                target = payload[key]
                break
        else:
            raise ValueError(f"Target dict at {path} does not contain a known tensor key.")
        metadata.pop(key, None)
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Target at {path} must be a tensor or tensor dict, got {type(target)}.")
    while target.ndim > 2 and target.size(0) == 1:
        target = target.squeeze(0)
    if target.ndim != 2:
        raise ValueError(f"Target at {path} must have shape [num_latents, hidden_size], got {tuple(target.shape)}.")
    return target.contiguous(), metadata


def _load_target_tensor(path: str) -> torch.Tensor:
    target, _ = _load_target_payload(path)
    return target


def _fit_target_len(
    target: torch.Tensor,
    num_tokens: int,
    target_resample_mode: str = "pool_avg",
) -> torch.Tensor:
    if target.size(0) == num_tokens:
        return target
    if target.size(0) == 0:
        raise ValueError("Cannot resize an empty latent target.")
    from src.train.monkey_patch_forward_univlr import (
        _adaptive_pool_sequence,
        build_univlr_target_resample_spec,
    )

    return _adaptive_pool_sequence(
        target,
        int(num_tokens),
        target_resample_spec=build_univlr_target_resample_spec(mode=target_resample_mode),
    )


class SupervisedDatasetUniVLRStage1(Dataset):
    """Stage-1 UniVLR dataset.

    This reader targets the atomic visual operation setting:
    user question/problem image -> one or more fixed-length visual latent segments -> final answer.
    It accepts either a UniVLR manifest with offline ``target_path`` fields, or
    Visual_CoT-style messages where assistant images are used as online visual targets.
    """

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
        padding: bool = True,
    ):
        super().__init__()
        if isinstance(data_path, str):
            if "," in data_path and not os.path.exists(data_path):
                list_data_dict: List[Dict[str, Any]] = []
                for path in data_path.split(","):
                    normalized_path = path.strip()
                    if not normalized_path:
                        continue
                    list_data_dict.extend(
                        _attach_manifest_metadata(_load_json_or_jsonl(normalized_path), normalized_path)
                    )
            else:
                list_data_dict = _attach_manifest_metadata(_load_json_or_jsonl(data_path), data_path)
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.target_folder = getattr(data_args, "univlr_target_folder", None)
        self.image_latent_tokens = int(getattr(data_args, "image_latent_tokens", 24))
        self.text_latent_tokens = int(getattr(data_args, "text_latent_tokens", 8))
        self.univlr_target_resample_mode = getattr(data_args, "univlr_target_resample_mode", "pool_avg")

        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.univlr_id = processor.tokenizer.convert_tokens_to_ids(UniVLR_TOKEN)
        self.univlr_end_id = processor.tokenizer.convert_tokens_to_ids(UniVLR_END_TOKEN)
        self.univlr_latent_end_id = processor.tokenizer.convert_tokens_to_ids(UniVLR_LATENT_END_TOKEN)
        self.mask_first_token_after_latent = bool(
            getattr(data_args, "mask_first_token_after_latent", False)
        )

    def __len__(self):
        return len(self.list_data_dict)

    def _load_image(self, image_path: str):
        image_path = _resolve_path(image_path, self.data_args.image_folder)
        return get_image_info(
            image_path,
            self.image_min_pixel,
            self.image_max_pixel,
            self.image_resized_w,
            self.image_resized_h,
        )

    def _resolve_target_path(self, path: str, sample: Optional[Dict[str, Any]] = None) -> str:
        if path is None or path == "":
            raise FileNotFoundError("UniVLR target file path is empty.")

        candidate_paths = []
        if os.path.exists(path):
            return path
        candidate_paths.append(path)

        manifest_dir = None
        if isinstance(sample, dict):
            manifest_dir = sample.get(MANIFEST_DIR_KEY)
        if manifest_dir:
            manifest_candidate = os.path.join(manifest_dir, path)
            if os.path.exists(manifest_candidate):
                return manifest_candidate
            candidate_paths.append(manifest_candidate)

        if self.target_folder:
            target_folder_candidate = os.path.join(self.target_folder, path)
            if os.path.exists(target_folder_candidate):
                return target_folder_candidate
            candidate_paths.append(target_folder_candidate)

        raise FileNotFoundError(
            "UniVLR target file not found. "
            f"path={path}, searched={candidate_paths}"
        )

    def _step_num_tokens(self, step: Dict[str, Any], kind: str) -> int:
        if "num_latents" in step:
            return int(step["num_latents"])
        if kind == "text":
            return int(step.get("num_text_latents", self.text_latent_tokens))
        if kind == "joint":
            return int(step.get("num_text_latents", self.text_latent_tokens)) + int(
                step.get("num_image_latents", self.image_latent_tokens)
            )
        return int(step.get("num_image_latents", self.image_latent_tokens))

    def _normalize_manifest(self, sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        system_text = sample.get("system", SYSTEM_MESSAGE)
        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})

        user_content: List[Dict[str, Any]] = []
        question = sample.get("question") or sample.get("Question") or sample.get("prompt") or ""
        problem_images = sample.get("problem_images") or sample.get("images") or []
        problem_image = sample.get("problem_image") or sample.get("problem_image_path") or sample.get("image")
        if problem_image:
            problem_images = [problem_image] + list(problem_images)
        for image_path in problem_images:
            user_content.append({"type": "image", "image": _resolve_path(image_path, self.data_args.image_folder)})
        if question:
            user_content.append({"type": "text", "text": str(question)})
        messages.append({"role": "user", "content": user_content})

        steps = sample.get("steps") or sample.get("latent_steps") or sample.get("cot_steps") or []
        answer = sample.get("answer") or sample.get("Final Answer") or sample.get("ground_truth") or ""
        return messages, steps, str(answer)

    def _normalize_visual_cot(self, sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        raw_messages = sample.get("data") or sample.get("messages")
        if raw_messages is None:
            raise ValueError("Visual_CoT sample must contain a 'data' or 'messages' field.")

        messages: List[Dict[str, Any]] = []
        steps: List[Dict[str, Any]] = []
        answer_parts: List[str] = []

        for message in raw_messages:
            role = message.get("role")
            content = copy.deepcopy(message.get("content", []))
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            if role == "assistant":
                for item in content:
                    item_type = item.get("type")
                    if item_type == "image":
                        steps.append(
                            {
                                "kind": "image",
                                "image": _resolve_path(item.get("image"), self.data_args.image_folder),
                                "num_latents": self.image_latent_tokens,
                            }
                        )
                    elif item_type == "text":
                        cleaned = _strip_answer_markup(item.get("text", ""))
                        if cleaned.strip():
                            answer_parts.append(cleaned)
                continue

            normalized_content = []
            for item in content:
                if item.get("type") == "image":
                    item["image"] = _resolve_path(item.get("image"), self.data_args.image_folder)
                normalized_content.append(item)
            messages.append({"role": role, "content": normalized_content})

        if not any(message.get("role") == "system" for message in messages) and SYSTEM_MESSAGE:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]})

        return messages, steps, "".join(answer_parts)

    def _normalize_step_manifest_with_messages(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        raw_messages = sample.get("data") or sample.get("messages")
        if raw_messages is None:
            raise ValueError("UniVLR manifest with explicit steps must contain a 'data' or 'messages' field.")

        messages: List[Dict[str, Any]] = []
        answer_parts: List[str] = []

        for message in raw_messages:
            role = message.get("role")
            content = copy.deepcopy(message.get("content", []))
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            # Explicit-step manifests already carry `steps`, so assistant entries
            # should not contribute additional target images.
            if role == "assistant":
                for item in content:
                    if item.get("type") == "text":
                        cleaned = _strip_answer_markup(item.get("text", ""))
                        if cleaned.strip():
                            answer_parts.append(cleaned)
                continue

            normalized_content = []
            for item in content:
                if item.get("type") == "image":
                    item["image"] = _resolve_path(item.get("image"), self.data_args.image_folder)
                normalized_content.append(item)
            messages.append({"role": role, "content": normalized_content})

        if not any(message.get("role") == "system" for message in messages) and SYSTEM_MESSAGE:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]})

        steps = sample.get("steps") or sample.get("latent_steps") or sample.get("cot_steps") or []
        answer = sample.get("answer") or sample.get("Final Answer") or sample.get("ground_truth") or "".join(answer_parts)
        return messages, steps, str(answer)

    def _normalize_llava_with_targets(self, sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        messages, steps, answer = self._normalize_manifest(sample)
        if steps:
            return messages, steps, answer
        raise ValueError(
            "LLaVA-style samples do not contain assistant target images. "
            "Use a UniVLR manifest with 'steps' and target_path/image fields for Stage1."
        )

    def _normalize_sample(self, sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        if ("steps" in sample or "latent_steps" in sample or "cot_steps" in sample) and (
            "data" in sample or "messages" in sample
        ):
            messages, steps, answer = self._normalize_step_manifest_with_messages(sample)
            if steps:
                return messages, steps, answer
        if "steps" in sample or "latent_steps" in sample or "cot_steps" in sample:
            messages, steps, answer = self._normalize_manifest(sample)
            if steps:
                return messages, steps, answer
        if "data" in sample or "messages" in sample:
            return self._normalize_visual_cot(sample)
        if "conversations" in sample:
            return self._normalize_llava_with_targets(sample)
        return self._normalize_manifest(sample)

    def _prepare_step_targets(self, steps: List[Dict[str, Any]], sample: Optional[Dict[str, Any]] = None):
        assistant_pieces: List[str] = []
        target_tensors: List[torch.Tensor] = []
        target_images = []
        latent_slot_types: List[int] = []
        latent_step_ids: List[int] = []
        latent_token_counts: List[int] = []

        for step_idx, step in enumerate(steps):
            kind = str(step.get("kind", "image")).lower()
            num_tokens = self._step_num_tokens(step, kind)
            assistant_pieces.append(_latent_segment(num_tokens))

            target_path = (
                step.get("target_path")
                or step.get("image_target_path")
                or step.get("text_target_path")
                or step.get("latent_target_path")
            )
            if target_path:
                resolved_target_path = self._resolve_target_path(target_path, sample=sample)
                tensor, metadata = _load_target_payload(resolved_target_path)
                payload_spec = metadata.get("target_resample_spec")
                if payload_spec is not None:
                    from src.train.monkey_patch_forward_univlr import build_univlr_target_resample_spec

                    expected_spec = build_univlr_target_resample_spec(
                        mode=self.univlr_target_resample_mode
                    )
                    payload_mode = build_univlr_target_resample_spec(
                        mode=payload_spec.get("mode") if isinstance(payload_spec, dict) else payload_spec
                    )
                    if payload_mode != expected_spec:
                        raise ValueError(
                            "Offline UniVLR target spec mismatch: "
                            f"file={resolved_target_path}, file_spec={payload_mode}, expected_spec={expected_spec}"
                        )
                tensor = _fit_target_len(
                    tensor,
                    num_tokens,
                    target_resample_mode=self.univlr_target_resample_mode,
                )
                target_tensors.append(tensor)
            else:
                image_path = step.get("image") or step.get("target_image") or step.get("image_path")
                if not image_path:
                    raise ValueError(f"Step {step_idx} has neither target_path nor image target.")
                target_images.append(self._load_image(image_path))
                latent_token_counts.append(num_tokens)

            if kind == "text":
                slot_types = [SLOT_TYPE_TEXT] * num_tokens
            elif kind == "joint":
                num_text = int(step.get("num_text_latents", self.text_latent_tokens))
                slot_types = [SLOT_TYPE_TEXT] * min(num_text, num_tokens)
                slot_types.extend([SLOT_TYPE_IMAGE] * (num_tokens - len(slot_types)))
            else:
                slot_types = [SLOT_TYPE_IMAGE] * num_tokens
            latent_slot_types.extend(slot_types)
            latent_step_ids.extend([step_idx] * num_tokens)

        if target_tensors and target_images:
            raise ValueError("A Stage1 batch item cannot mix offline target_path and online target images.")

        return assistant_pieces, target_tensors, target_images, latent_token_counts, latent_slot_types, latent_step_ids

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        messages, steps, answer = self._normalize_sample(sample)
        if not steps:
            raise ValueError(f"UniVLR Stage1 sample {i} has no latent target step.")

        (
            assistant_pieces,
            target_tensors,
            target_images,
            latent_token_counts,
            latent_slot_types,
            latent_step_ids,
        ) = self._prepare_step_targets(steps, sample=sample)

        assistant_text = "".join(assistant_pieces) + UniVLR_LATENT_END_TOKEN + answer
        full_messages = copy.deepcopy(messages)
        full_messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        user_images, user_videos = process_vision_info(full_messages)
        user_images = user_images if user_images else None
        user_videos = user_videos if user_videos else None

        inputs = self.processor(
            text=[full_text],
            images=user_images,
            videos=user_videos,
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=user_images,
            videos=user_videos,
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0).to(torch.long)
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = IGNORE_INDEX
        labels[labels == self.univlr_id] = IGNORE_INDEX
        if self.mask_first_token_after_latent:
            if self.univlr_latent_end_id is not None and self.univlr_latent_end_id >= 0:
                latent_end_positions = torch.nonzero(input_ids == self.univlr_latent_end_id, as_tuple=True)[0]
            else:
                latent_end_positions = torch.nonzero(input_ids == self.univlr_end_id, as_tuple=True)[0]
            first_after_latent = latent_end_positions + 1
            first_after_latent = first_after_latent[first_after_latent < labels.shape[0]]
            labels[first_after_latent] = IGNORE_INDEX

        latent_slot_types_tensor = torch.tensor(latent_slot_types, dtype=torch.long)
        latent_step_ids_tensor = torch.tensor(latent_step_ids, dtype=torch.long)
        num_univlr_tokens = int((input_ids == self.univlr_id).sum().item())
        if num_univlr_tokens != latent_slot_types_tensor.numel():
            raise ValueError(
                f"Sample {i} has {num_univlr_tokens} <|univlr|> tokens but "
                f"{latent_slot_types_tensor.numel()} latent target slots."
            )

        data_dict: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            "latent_slot_types": latent_slot_types_tensor,
            "latent_step_ids": latent_step_ids_tensor,
        }

        if "pixel_values" in inputs:
            data_dict["pixel_values"] = inputs["pixel_values"]
            data_dict["image_grid_thw"] = inputs["image_grid_thw"]
        if "pixel_values_videos" in inputs:
            data_dict["pixel_values_videos"] = inputs["pixel_values_videos"]
            data_dict["video_grid_thw"] = inputs["video_grid_thw"]
        if "second_per_grid_ts" in inputs:
            data_dict["second_per_grid_ts"] = inputs["second_per_grid_ts"]

        if target_tensors:
            data_dict["latent_targets"] = torch.cat(target_tensors, dim=0)
        else:
            target_texts = [f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}"] * len(target_images)
            target_inputs = self.processor(
                text=target_texts,
                images=target_images,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            data_dict["univlr_pixel_values"] = target_inputs["pixel_values"]
            data_dict["univlr_image_grid_thw"] = target_inputs["image_grid_thw"]
            data_dict["univlr_token_counts"] = torch.tensor(latent_token_counts, dtype=torch.long)

        return data_dict


class DataCollatorForUniVLRStage1(object):
    def __init__(self, pad_token_id: int, univlr_token_id: int):
        self.pad_token_id = pad_token_id
        self.univlr_token_id = univlr_token_id

    def __call__(self, examples):
        input_ids = pad_sequence(
            [example["input_ids"] for example in examples],
            padding_side="right",
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [example["labels"] for example in examples],
            padding_side="right",
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids != self.pad_token_id

        latent_slot_types = torch.cat([example["latent_slot_types"] for example in examples], dim=0)
        latent_step_ids = torch.cat([example["latent_step_ids"] for example in examples], dim=0)
        latent_sample_ids = torch.cat(
            [
                torch.full_like(example["latent_slot_types"], fill_value=batch_idx)
                for batch_idx, example in enumerate(examples)
            ],
            dim=0,
        )

        data_dict: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "latent_slot_types": latent_slot_types,
            "latent_step_ids": latent_step_ids,
            "latent_sample_ids": latent_sample_ids,
        }

        if sum(int((example["input_ids"] == self.univlr_token_id).sum().item()) for example in examples) != latent_slot_types.numel():
            raise ValueError("Batch <|univlr|> token count does not match latent metadata length.")

        if any("pixel_values" in example for example in examples):
            data_dict["pixel_values"] = torch.cat(
                [example["pixel_values"] for example in examples if "pixel_values" in example],
                dim=0,
            )
            data_dict["image_grid_thw"] = torch.cat(
                [example["image_grid_thw"] for example in examples if "image_grid_thw" in example],
                dim=0,
            )
        if any("pixel_values_videos" in example for example in examples):
            data_dict["pixel_values_videos"] = torch.cat(
                [example["pixel_values_videos"] for example in examples if "pixel_values_videos" in example],
                dim=0,
            )
            data_dict["video_grid_thw"] = torch.cat(
                [example["video_grid_thw"] for example in examples if "video_grid_thw" in example],
                dim=0,
            )
        if any("second_per_grid_ts" in example for example in examples):
            second_per_grid_ts = []
            for example in examples:
                if "second_per_grid_ts" in example:
                    second_per_grid_ts.extend(example["second_per_grid_ts"])
            data_dict["second_per_grid_ts"] = second_per_grid_ts

        has_offline_targets = any("latent_targets" in example for example in examples)
        has_online_targets = any("univlr_pixel_values" in example for example in examples)
        if has_offline_targets and has_online_targets:
            raise ValueError("Do not mix offline latent_targets and online assistant-image targets in one batch.")
        if has_offline_targets:
            data_dict["latent_targets"] = torch.cat([example["latent_targets"] for example in examples], dim=0)
        elif has_online_targets:
            data_dict["univlr_pixel_values"] = torch.cat(
                [example["univlr_pixel_values"] for example in examples],
                dim=0,
            )
            data_dict["univlr_image_grid_thw"] = torch.cat(
                [example["univlr_image_grid_thw"] for example in examples],
                dim=0,
            )
            data_dict["univlr_token_counts"] = torch.cat(
                [example["univlr_token_counts"] for example in examples],
                dim=0,
            )

        return data_dict


def make_supervised_data_module_univlr_stage1(model_id, processor, data_args):
    sft_dataset = SupervisedDatasetUniVLRStage1(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )
    univlr_token_id = processor.tokenizer.convert_tokens_to_ids(UniVLR_TOKEN)
    data_collator = DataCollatorForUniVLRStage1(
        pad_token_id=processor.tokenizer.pad_token_id,
        univlr_token_id=univlr_token_id,
    )
    return dict(train_dataset=sft_dataset, eval_dataset=None, data_collator=data_collator)
