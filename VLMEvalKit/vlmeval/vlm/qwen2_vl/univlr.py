from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor

from vlmeval.smp import get_cache_path
from ..base import BaseModel
from .model import ensure_image_url, ensure_video_url
from .prompt import Qwen2VLPromptMixin


DEFAULT_UNIVLR_ROOT = str(Path(__file__).resolve().parents[4])


def _ensure_univlr_root() -> str:
    univlr_root = os.environ.get('UNIVLR_ROOT') or DEFAULT_UNIVLR_ROOT
    if not os.path.isdir(univlr_root):
        raise FileNotFoundError(
            f'UNIVLR_ROOT={univlr_root} does not exist. Set UNIVLR_ROOT to the UniVLR repository root.'
        )
    if univlr_root not in sys.path:
        sys.path.insert(0, univlr_root)
    return univlr_root


def _as_univlr_steps(value: int | str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    return [int(x) for x in value]


def _first_univlr_step(univlr_steps: list[int] | None) -> int | None:
    if not univlr_steps:
        return None
    return int(univlr_steps[0])


def _decode_univlr_output(
    token_ids: torch.Tensor,
    processor: Any,
    config: Any,
    decoding_strategy: str | None = None,
    univlr_steps: list[int] | None = None,
) -> str:
    tokenizer = getattr(processor, 'tokenizer', processor)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    rendered_ids = []
    in_univlr_span = False
    after_latent_end = False
    univlr_span_count = 0
    fixed_univlr_steps = None
    if decoding_strategy == 'steps' and univlr_steps:
        fixed_univlr_steps = _first_univlr_step(univlr_steps)

    def close_univlr_span():
        nonlocal in_univlr_span
        if in_univlr_span:
            in_univlr_span = False
            if not rendered_ids or rendered_ids[-1] != config.univlr_end_id:
                rendered_ids.append(config.univlr_end_id)

    for token_id in token_ids.tolist():
        token_id = int(token_id)
        if after_latent_end and token_id in {
            im_end_id,
            getattr(config, 'univlr_start_id', -1),
            getattr(config, 'univlr_id', -1),
            getattr(config, 'univlr_end_id', -1),
            getattr(config, 'univlr_latent_end_id', -1),
        }:
            continue
        if token_id == im_end_id:
            if in_univlr_span and fixed_univlr_steps is not None and univlr_span_count >= fixed_univlr_steps:
                close_univlr_span()
            continue
        if token_id == config.univlr_end_id:
            in_univlr_span = False
            if not rendered_ids or rendered_ids[-1] != config.univlr_end_id:
                rendered_ids.append(token_id)
            continue
        if token_id == getattr(config, 'univlr_latent_end_id', -1):
            close_univlr_span()
            rendered_ids.append(token_id)
            after_latent_end = True
            continue

        if in_univlr_span and fixed_univlr_steps is not None and univlr_span_count >= fixed_univlr_steps:
            close_univlr_span()

        if token_id == config.univlr_start_id:
            in_univlr_span = True
            univlr_span_count = 0
            rendered_ids.append(token_id)
            continue

        if in_univlr_span:
            rendered_ids.append(config.univlr_id)
            univlr_span_count += 1
            continue

        rendered_ids.append(token_id)

    if in_univlr_span and fixed_univlr_steps is not None and univlr_span_count >= fixed_univlr_steps:
        close_univlr_span()

    return tokenizer.decode(
        rendered_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _clean_univlr_response(response: str) -> str:
    response = re.sub(r'<\|univlr_start\|>.*?<\|univlr_end\|>', '', response, flags=re.DOTALL)
    for token in (
        '<|univlr_start|>',
        '<|univlr_end|>',
        '<|univlr|>',
        '<|univlr_latent_end|>',
        '<|im_start|>',
        '<|im_end|>',
        '<|endoftext|>',
    ):
        response = response.replace(token, '')
    return response.strip()


class Qwen2VLChatUniVLR(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 512,
        decoding_strategy: str = 'univlr',
        univlr_steps: int | str | list[int] | tuple[int, ...] = 1,
        criterion: str | None = None,
        univlr_end_threshold: float | None = None,
        torch_dtype: str = 'auto',
        attn_implementation: str = 'flash_attention_2',
        device_map: str | None = None,
        trust_remote_code: bool = True,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        clean_univlr_output: bool = True,
        post_process: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        if kwargs.get('use_vllm', False):
            raise ValueError('Qwen2VLChatUniVLR does not support vLLM; run without --use-vllm.')

        _ensure_univlr_root()
        from src.model.qwen_univlr_model import QwenWithUniVLR
        from src.train.monkey_patch_forward_univlr import replace_qwen2_5_with_mixed_modality_forward_univlr

        if not os.path.exists(model_path):
            cache_path = get_cache_path(model_path, repo_type='models')
            if cache_path is None:
                snapshot_download(repo_id=model_path)
                cache_path = get_cache_path(model_path, repo_type='models')
            model_path = cache_path

        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.decoding_strategy = decoding_strategy
        self.univlr_steps = _as_univlr_steps(univlr_steps)
        self.criterion = criterion
        self.univlr_end_threshold = univlr_end_threshold
        self.system_prompt = system_prompt
        self.clean_univlr_output = clean_univlr_output
        self.post_process = post_process
        self.verbose = verbose

        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        replace_qwen2_5_with_mixed_modality_forward_univlr(
            inference_mode=True,
            univlr_head=getattr(self.config, 'univlr_head', True),
        )

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)
            resolved_device_map: str | dict[str, str]
            if device_map is None or device_map == 'rank_local':
                resolved_device_map = {'': str(self.device)}
            elif device_map == 'auto':
                resolved_device_map = 'auto'
            elif isinstance(device_map, str) and device_map.startswith('cuda'):
                resolved_device_map = {'': device_map}
            else:
                resolved_device_map = device_map
        else:
            self.device = torch.device('cpu')
            resolved_device_map = {'': 'cpu'}

        self.model = QwenWithUniVLR.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=resolved_device_map,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, Any]]:
        content = []
        for item in inputs:
            if item['type'] == 'image':
                image_item = {'type': 'image', 'image': ensure_image_url(item['value'])}
                if self.min_pixels is not None:
                    image_item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    image_item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    image_item['total_pixels'] = self.total_pixels
                content.append(image_item)
            elif item['type'] == 'video':
                video_item = {'type': 'video', 'video': ensure_video_url(item['value'])}
                if self.min_pixels is not None:
                    video_item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    video_item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    video_item['total_pixels'] = self.total_pixels
                content.append(video_item)
            elif item['type'] == 'text':
                content.append({'type': 'text', 'text': item['value']})
            else:
                raise ValueError(f"Invalid message type: {item['type']}, {item}")
        return content

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'decoding_strategy': self.decoding_strategy,
            'univlr_steps': self.univlr_steps,
        }
        if self.criterion is not None:
            kwargs['criterion'] = self.criterion
        if self.univlr_end_threshold is not None:
            kwargs['univlr_end_threshold'] = self.univlr_end_threshold
        return kwargs

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        if self.verbose:
            print(f'\033[31m{messages}\033[0m', flush=True)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors='pt',
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = _decode_univlr_output(
                generated_ids[0],
                self.processor,
                self.model.config,
                decoding_strategy=self.decoding_strategy,
                univlr_steps=self.univlr_steps,
            )

        if self.clean_univlr_output:
            response = _clean_univlr_response(response)

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            counter, end = 1, None
            for i, char in enumerate(resp):
                if char == '{':
                    counter += 1
                elif char == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                if i == len(resp) - 1:
                    end = len(resp)
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m', flush=True)
        return response
