"""
    Implementation of UniVLR models based on Qwen-2.5-VL series
"""
import math
import torch.nn as nn
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.configuration_utils import PretrainedConfig

from torch.nn import CrossEntropyLoss, MSELoss, L1Loss

import inspect
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.nn import functional as F


from transformers.generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerationConfig,
    GenerationMode,
)

from transformers.generation.logits_process import (
    LogitsProcessorList,
)


from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    logger,
    GenerateNonBeamOutput,
    GenerateOutput,
    GenerateEncoderDecoderOutput, 
    GenerateDecoderOnlyOutput,
)

from transformers.generation.streamers import BaseStreamer
from transformers.cache_utils import Cache
from transformers import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled  
from transformers.integrations.fsdp import is_fsdp_managed_module


from src.model.univlr_heads import UniVLRHead, UniVLRHeadGLU, UniVLRHeadPrior, UniVLRHeadVAE

class QwenWithUniVLR(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.univlr_head:
            self._init_univlr_head(config.univlr_head_type)
        if config.latent_end_token:
            self._init_univlr_latent_end_emb()
        
    def _init_univlr_head(self,univlr_head_type):
        print(f"Detected UniVLR Head Type: '{univlr_head_type}'")
        if hasattr(self, "univlr_head_prior"):
            del self.univlr_head_prior
        if univlr_head_type == 'simple':
            self.univlr_head = UniVLRHead(hidden_size=self.config.hidden_size)
        elif univlr_head_type == 'glu':
            self.univlr_head = UniVLRHeadGLU(hidden_size=self.config.hidden_size, 
                                       intermediate_size=self.config.intermediate_size,
                                       hidden_act=self.config.hidden_act)
        elif univlr_head_type == 'vae':
            logvar_min = float(getattr(self.config, "univlr_vae_logvar_min", -6.0))
            logvar_max = float(getattr(self.config, "univlr_vae_logvar_max", 2.0))
            prior_type = str(getattr(self.config, "univlr_vae_prior_type", "learned")).lower()
            self.univlr_head = UniVLRHeadVAE(
                hidden_size=self.config.hidden_size,
                logvar_min=logvar_min,
                logvar_max=logvar_max,
            )
            if prior_type == "learned":
                self.univlr_head_prior = UniVLRHeadPrior(
                    hidden_size=self.config.hidden_size,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                )
            elif prior_type not in {"target", "target_detached", "identity", "fixed", "standard", "standard_normal", "gaussian"}:
                raise ValueError(
                    f"Unknown univlr_vae_prior_type: '{prior_type}'. "
                    "Supported variants are 'learned', 'target', and 'standard'."
                )
        else:
            # Raise an error for an unknown variant to prevent silent failures
            raise ValueError(f"Unknown univlr_head_type: '{univlr_head_type}'. "
                             "Supported variants are 'simple', 'glu', 'vae'.")
        self.config.univlr_head_type = univlr_head_type

    def is_variational_univlr(self) -> bool:
        return bool(getattr(self.config, "univlr_head", False)) and str(
            getattr(self.config, "univlr_head_type", "")
        ).lower() == "vae"

    def predict_univlr_distribution(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not getattr(self.config, "univlr_head", False):
            return hidden_states, torch.zeros_like(hidden_states)
        if not hasattr(self, "univlr_head"):
            raise ValueError("config.univlr_head=True, but model has no univlr_head module.")
        if hasattr(self.univlr_head, "distribution"):
            return self.univlr_head.distribution(hidden_states)
        projected = self.univlr_head(hidden_states)
        return projected, torch.zeros_like(projected)

    def predict_univlr_prior_distribution(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_variational_univlr():
            return hidden_states, torch.zeros_like(hidden_states)
        prior_type = str(getattr(self.config, "univlr_vae_prior_type", "learned")).lower()
        target_logvar = float(getattr(self.config, "univlr_vae_target_logvar", 0.0))
        if prior_type == "learned":
            if not hasattr(self, "univlr_head_prior"):
                raise ValueError("univlr_head_type='vae' with univlr_vae_prior_type='learned' requires model.univlr_head_prior.")
            return self.univlr_head_prior.distribution(hidden_states)
        if prior_type in {"target", "target_detached", "identity", "fixed"}:
            return hidden_states, torch.full_like(hidden_states, target_logvar)
        if prior_type in {"standard", "standard_normal", "gaussian"}:
            return torch.zeros_like(hidden_states), torch.full_like(hidden_states, target_logvar)
        raise ValueError(
            f"Unknown univlr_vae_prior_type: '{prior_type}'. "
            "Supported variants are 'learned', 'target', and 'standard'."
        )

        
    def _init_univlr_latent_end_emb(self):

        print(f"Activated Learnable latent end token of UniVLR")
        # Initializing the learnable latentend
        # 2X norm to distinguish this from the normal semantic space
        target_norm_scale_latentend = 1
        with torch.no_grad():
            v = torch.randn(self.config.hidden_size, dtype=self.dtype, device=self.device)
            v = v / (v.norm() + 1e-6)
            v = v * (target_norm_scale_latentend * math.sqrt(self.config.hidden_size))
        self.univlr_latent_end_emb = torch.nn.Parameter(v)

        # univlr_latent_end_emb = torch.full(
        #     (config.hidden_size,),
        #     fill_value=1.0 / self.config.hidden_size
        # )
        # self.register_buffer('univlr_latent_end_emb', univlr_latent_end_emb)    # will not compute grad
    
    # Patch the generation function with univlr_generate
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        decoding_strategy:Optional[str]=None,
        criterion: Optional[str]="mse",
        univlr_end_threshold: Optional[float]=0.02,
        univlr_steps: Optional[List[int]]=None,
        univlr_block_size: Optional[int]=None,
        **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
        """
            Patching the generation function for UniVLR
        """

        # Params in 
        if decoding_strategy is None and hasattr(generation_config,'decoding_strategy'):
            decoding_strategy = generation_config.decoding_strategy
        if criterion is None and hasattr(generation_config,'criterion'):
            criterion = generation_config.criterion
        if univlr_end_threshold is None and hasattr(generation_config,'univlr_end_threshold'):
            univlr_end_threshold = generation_config.univlr_end_threshold
        if univlr_steps is None and hasattr(generation_config,'univlr_steps'):
            univlr_steps = generation_config.univlr_steps
        if univlr_block_size is None and hasattr(generation_config, 'univlr_block_size'):
            univlr_block_size = generation_config.univlr_block_size
        if univlr_block_size is None:
            univlr_block_size = kwargs.pop("univlr_block_size", None)

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        # self._validate_model_kwargs()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        # generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        '''
            No other modes
            UniVLR decoding only
        '''
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        '''
            ._sample is patched by _univlr_decoding
        '''
        if decoding_strategy == "latent":
            result = self._univlr_deocding_with_latentend(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                criterion = criterion,
                univlr_end_threshold= univlr_end_threshold,
                univlr_max_steps = univlr_steps,
                **model_kwargs,)
            
        elif decoding_strategy == "steps":
            result = self._univlr_deocding_by_steps(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                univlr_steps = univlr_steps,
                **model_kwargs,)
        elif decoding_strategy == "univlr":
            result = self._univlr_deocding_univlr(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                univlr_steps=univlr_steps,
                univlr_block_size=univlr_block_size,
                **model_kwargs,
            )
        else:
            # Vanilla decoding
            # enters UniVLR if it sees start
            # exits UniVLR if it sees end
            result = self._univlr_deocding(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                criterion = criterion,
                univlr_end_threshold= univlr_end_threshold,
                univlr_steps=univlr_steps,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result
    
    # UniVLR docoding logic
    def _univlr_deocding(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional["BaseStreamer"],
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                    `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            pad_token_id = generation_config._pad_token_tensor
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            
            # model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
            # again Transformer version issue
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

            model_forward = self.__call__
            if isinstance(model_kwargs.get("past_key_values"), Cache):
                is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
                if getattr(self, "hf_quantizer", None) is not None:
                    is_compileable &= self.hf_quantizer.is_compileable
                is_compileable = is_compileable and not generation_config.disable_compile
                if is_compileable and (
                    self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
                ):
                    os.environ["TOKENIZERS_PARALLELISM"] = "0"
                    model_forward = self.get_compiled_call(generation_config.compile_config)

            if generation_config.prefill_chunk_size is not None:
                model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
                is_prefill = False
            else:
                is_prefill = True

            univlr_mode_switch = torch.zeros(batch_size,dtype=torch.bool,device=input_ids.device)  # switch gate for univlr mode
            last_position_hidden_state = None
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
                
                model_inputs.update({"univlr_mode_switch":univlr_mode_switch})
                model_inputs.update({"last_position_hidden_state":last_position_hidden_state})
                if is_prefill:
                    outputs = self(**model_inputs, return_dict=True)
                    is_prefill = False
                else:
                    outputs = model_forward(**model_inputs, return_dict=True)

                # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                if synced_gpus and this_peer_finished:
                    continue

                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                '''
                    UniVLR reasoning mode switches:

                    When next token is <|univlr_start|>, we still need to pass its token id through decoding
                    When last token is <|univlr_start|>, we will start passing hidden states (enter univlr mode)

                    When next token is <|univlr_end|>, we will stop passing hidden states (end univlr mode)

                    ONE ASSUMPTION: The UniVLR hidden states shall not trigger <|univlr_end|>
                '''
                last_tokens = input_ids[:,-1]
                univlr_start_switch = (last_tokens == self.config.univlr_start_id).to(device=input_ids.device)            
                univlr_end_switch = (next_tokens == self.config.univlr_end_id).to(device=input_ids.device)                
                '''
                    Goal: univlr_mode_switch = univlr_mode_switch + univlr_start_switch - univlr_end_switch

                '''
                univlr_mode_switch = (univlr_mode_switch | univlr_start_switch) & (~univlr_end_switch)
                last_position_hidden_state = outputs.last_position_hidden_state

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                    
                # unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                """if univlr mode is unfinished, do not stop"""
                unfinished_sequences = (
                    univlr_mode_switch | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
                )
                this_peer_finished = unfinished_sequences.max() == 0
                cur_len += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

            if streamer is not None:
                streamer.end()

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids

    def _univlr_deocding_with_latentend(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional["BaseStreamer"],
            criterion: Optional[str]="mse",
            univlr_end_threshold: Optional[float]=0.1,
            univlr_max_steps: Optional[int]=16,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                    `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            univlr_max_steps = univlr_max_steps[0] if isinstance(univlr_max_steps,list) else univlr_max_steps
            pad_token_id = generation_config._pad_token_tensor
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            
            model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
            # model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

            model_forward = self.__call__
            if isinstance(model_kwargs.get("past_key_values"), Cache):
                is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
                if getattr(self, "hf_quantizer", None) is not None:
                    is_compileable &= self.hf_quantizer.is_compileable
                is_compileable = is_compileable and not generation_config.disable_compile
                if is_compileable and (
                    self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
                ):
                    os.environ["TOKENIZERS_PARALLELISM"] = "0"
                    model_forward = self.get_compiled_call(generation_config.compile_config)

            if generation_config.prefill_chunk_size is not None:
                model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
                is_prefill = False
            else:
                is_prefill = True

            # print(criterion)
            
            '''set the univlr latent end criterion'''
            if criterion == 'mse':
                criterion_fct = MSELoss(reduction="none")
            elif criterion == 'mae':
                criterion_fct = L1Loss(reduction="none")
            elif criterion == 'cosine':
                # Returns a loss function: 1 - cosine similarity
                def cosine_loss(x, y):
                    return 1 - F.cosine_similarity(x, y, dim=-1).mean()
                criterion_fct = cosine_loss
            else:
                raise ValueError(f"Unsupported criterion_fct: {criterion_fct}")

            univlr_mode_switch = torch.zeros(batch_size,dtype=torch.bool,device=input_ids.device)  # switch gate for univlr mode
            last_position_hidden_state = None

            univlr_step_counter = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device) # track UniVLR steps
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
                
                model_inputs.update({"univlr_mode_switch":univlr_mode_switch})
                model_inputs.update({"last_position_hidden_state":last_position_hidden_state})

                if is_prefill:
                    outputs = self(**model_inputs, return_dict=True)
                    is_prefill = False
                else:
                    outputs = model_forward(**model_inputs, return_dict=True)

                # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                if synced_gpus and this_peer_finished:
                    continue

                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                '''
                    UniVLR reasoning mode switches:

                    When next token is <|univlr_start|>, we still need to pass its token id through decoding
                    When last token is <|univlr_start|>, we will start passing hidden states (enter univlr mode)

                    When current output's last_hidden_state ~ univlr_latent_end, we still pass hidden states 
                    When last position's last_hidden_state ~ univlr_latent_end, we will start passing tokens(end univlr mode)

                    ONE ASSUMPTION: The UniVLR hidden states shall not trigger <|univlr_end|>
                '''

                last_tokens = input_ids[:,-1]
                univlr_start_switch = (last_tokens == self.config.univlr_start_id).to(device=input_ids.device)            
                '''
                    At this moment, the last_position_hidden_states has not been updated,
                    it is still the output of the position before.
                    We check 
                '''
                if last_position_hidden_state is None:
                    univlr_end_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
                else:
                    univlr_end_switch = criterion_fct(
                                        last_position_hidden_state, 
                                        self.univlr_latent_end_emb.unsqueeze(0).expand_as(last_position_hidden_state)
                                        ).mean(dim=-1)
                    univlr_end_switch = univlr_end_switch < univlr_end_threshold

                # If currently in UniVLR mode, increment counter
                univlr_step_counter = univlr_step_counter + univlr_mode_switch.long()
                # Reset counter when entering UniVLR mode
                univlr_step_counter = torch.where(univlr_start_switch, torch.zeros_like(univlr_step_counter), univlr_step_counter)
                # Force exit if budget exceeded
                univlr_budget_exceeded = univlr_step_counter >= univlr_max_steps

                '''
                    Goal: univlr_mode_switch = univlr_mode_switch + univlr_start_switch - univlr_end_switch
                    Update: exit univlr when univlr_budget_exceeded 

                '''
                # univlr_mode_switch = ((univlr_mode_switch | univlr_start_switch) & (~univlr_end_switch)).to(torch.bool)
                univlr_mode_switch = ((univlr_mode_switch | univlr_start_switch) & (~univlr_end_switch) & (~univlr_budget_exceeded)).to(torch.bool)

                last_position_hidden_state = outputs.last_position_hidden_state     #We can now update the last position hidden states    

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                # unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                """if univlr mode is unfinished, do not stop"""
                unfinished_sequences = (
                    univlr_mode_switch | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
                )
                # print(univlr_mode_switch)
                this_peer_finished = unfinished_sequences.max() == 0
                cur_len += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

            if streamer is not None:
                streamer.end()

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids

# UniVLR docoding logic
    def _univlr_deocding_by_steps(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        univlr_steps: List[int],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        try:
            model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        except:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        univlr_mode_switch = torch.zeros(batch_size,dtype=torch.bool,device=input_ids.device)  # switch gate for univlr mode
        last_position_hidden_state = None

        # Track UniVLR quotas
        univlr_steps_orig = torch.tensor(univlr_steps, dtype=torch.int, device=input_ids.device)  # original quota
        univlr_remaining_steps = univlr_steps_orig.clone()
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            
            model_inputs.update({"univlr_mode_switch":univlr_mode_switch})
            model_inputs.update({"last_position_hidden_state":last_position_hidden_state})
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            '''
                UniVLR reasoning mode switches:

                When next token is <|univlr_start|>, we still need to pass its token id through decoding
                When last token is <|univlr_start|>, we will start passing hidden states (enter univlr mode)

                During UniVLR, keep passing hidden_states until quota uses up
            '''
            last_tokens = input_ids[:,-1]
            univlr_start_switch = (last_tokens == self.config.univlr_start_id).to(device=input_ids.device)            

            # UniVLR stage-1 is trained on fixed spans:
            # <univlr_start> + N latent slots + <univlr_end>. The latent-slot token
            # logits themselves are not supervised, so do not rely on the LM to
            # naturally emit the boundary after the fixed quota is consumed.
            univlr_end_id = getattr(self.config, "univlr_end_id", None)
            if univlr_end_id is not None and int(univlr_end_id) >= 0:
                force_univlr_end = univlr_mode_switch & (univlr_remaining_steps <= 1)
                next_tokens = torch.where(
                    force_univlr_end,
                    torch.full_like(next_tokens, int(univlr_end_id)),
                    next_tokens,
                )

            '''
                Goal: univlr_mode_switch = univlr_mode_switch + univlr_start_switch 
                the exit is controlled by univlr quota now, not <|univlr_end|>

            '''
            # Candidate new switch (no end token anymore)
            new_mode_switch = univlr_mode_switch | univlr_start_switch

            # Detect entry vs continuation
            just_entered = (~univlr_mode_switch) & new_mode_switch
            still_in     = univlr_mode_switch & new_mode_switch

            # Reset quota when entering
            univlr_remaining_steps = torch.where(just_entered, univlr_steps_orig, univlr_remaining_steps)

            # Decrement quota only if we were already inside before this step
            univlr_remaining_steps = univlr_remaining_steps - univlr_mode_switch.long()

            # Exit if quota used up
            univlr_mode_switch = new_mode_switch & (univlr_remaining_steps > 0)


            last_position_hidden_state = outputs.last_position_hidden_state

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            """if univlr mode is unfinished, do not stop"""
            unfinished_sequences = (
                univlr_mode_switch | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
            )

            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _univlr_deocding_univlr(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        univlr_steps: List[int],
        univlr_block_size: Optional[int] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """Decode structured UniVLR chains.

        ``univlr_steps`` is interpreted as the number of latent blocks to emit for
        each sample. Each block contains ``univlr_block_size`` effective
        latent slots, rendered as:

            <|univlr_start|> 24 * <|univlr|> <|univlr_end|>

        After the final block, decoding forces ``<|univlr_latent_end|>`` and then
        switches back to normal token generation for the answer text.
        """
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        try:
            model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        except Exception:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        if univlr_steps is None:
            raise ValueError("decoding_strategy='univlr' requires univlr_steps to specify the number of latent blocks.")
        if isinstance(univlr_steps, int):
            univlr_steps = [univlr_steps]
        univlr_steps = [int(x) for x in univlr_steps]
        if len(univlr_steps) == 1 and batch_size > 1:
            univlr_steps = univlr_steps * batch_size
        if len(univlr_steps) != batch_size:
            raise ValueError(
                f"univlr decoding expected {batch_size} univlr_steps values, got {len(univlr_steps)}: {univlr_steps}"
            )

        slots_per_block = univlr_block_size
        if slots_per_block is None:
            slots_per_block = getattr(self.config, "univlr_slots_per_block", None)
        if slots_per_block is None:
            slots_per_block = getattr(self.config, "image_latent_tokens", 24)
        slots_per_block = int(slots_per_block)
        if slots_per_block <= 0:
            raise ValueError(f"univlr_block_size must be positive, got {slots_per_block}")

        univlr_start_id = int(getattr(self.config, "univlr_start_id"))
        univlr_end_id = int(getattr(self.config, "univlr_end_id"))
        univlr_latent_end_id = getattr(self.config, "univlr_latent_end_id", None)
        if univlr_latent_end_id is None or int(univlr_latent_end_id) < 0:
            raise ValueError("decoding_strategy='univlr' requires config.univlr_latent_end_id to be set.")
        univlr_latent_end_id = int(univlr_latent_end_id)

        univlr_mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        last_position_hidden_state = None
        target_blocks = torch.tensor(univlr_steps, dtype=torch.long, device=input_ids.device)
        blocks_completed = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        slots_remaining = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            model_inputs.update({"univlr_mode_switch": univlr_mode_switch})
            model_inputs.update({"last_position_hidden_state": last_position_hidden_state})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            last_tokens = input_ids[:, -1]
            univlr_start_switch = (last_tokens == univlr_start_id) & (blocks_completed < target_blocks)

            force_univlr_end = univlr_mode_switch & (slots_remaining <= 1)
            if bool(force_univlr_end.any().item()):
                next_tokens = torch.where(force_univlr_end, torch.full_like(next_tokens, univlr_end_id), next_tokens)

            after_block_end = last_tokens == univlr_end_id
            force_next_block = after_block_end & (blocks_completed < target_blocks)
            if bool(force_next_block.any().item()):
                next_tokens = torch.where(force_next_block, torch.full_like(next_tokens, univlr_start_id), next_tokens)

            force_latent_end = after_block_end & (blocks_completed >= target_blocks) & (target_blocks > 0)
            if bool(force_latent_end.any().item()):
                next_tokens = torch.where(
                    force_latent_end,
                    torch.full_like(next_tokens, univlr_latent_end_id),
                    next_tokens,
                )

            new_mode_switch = (univlr_mode_switch | univlr_start_switch).to(torch.bool)
            just_entered = (~univlr_mode_switch) & new_mode_switch
            slots_remaining = torch.where(
                just_entered,
                torch.full_like(slots_remaining, slots_per_block),
                slots_remaining,
            )
            slots_remaining = slots_remaining - univlr_mode_switch.long()

            emitted_univlr_end = next_tokens == univlr_end_id
            blocks_completed = blocks_completed + emitted_univlr_end.long()
            univlr_mode_switch = new_mode_switch & (~emitted_univlr_end) & (slots_remaining > 0)

            last_position_hidden_state = outputs.last_position_hidden_state
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = (
                univlr_mode_switch
                | force_next_block
                | force_latent_end
                | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    # @classmethod
    # def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
    #     """
    #     Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`]. This function is useful to convert legacy
    #     [`PretrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

    #     Args:
    #         model_config (`PretrainedConfig`):
    #             The model config that will be used to instantiate the generation config.

    #     Returns:
    #         [`GenerationConfig`]: The configuration object instantiated from those parameters.
    #     """
    #     config_dict = model_config.to_dict()
    #     config_dict.pop("_from_model_config", None)

    #     # Removes all `None` from the model config dict -- this lets the generation config defaults to take hold
    #     config_dict = {key: value for key, value in config_dict.items() if value is not None}

    #     generation_config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

    #     # Special case: some models have generation attributes set in the decoder. Use them if still unset in the
    #     # generation config (which in turn is defined from the outer attributes of model config).
    #     decoder_config = model_config.get_text_config(decoder=True)
    #     if decoder_config is not model_config:
    #         default_generation_config = GenerationConfig()
    #         decoder_config_dict = decoder_config.to_dict() if isinstance(decoder_config, PretrainedConfig) else decoder_config

    #         for attr in generation_config.to_dict().keys():
    #             is_unset = getattr(generation_config, attr) == getattr(default_generation_config, attr)
    #             if attr in decoder_config_dict and is_unset:
    #                 setattr(generation_config, attr, decoder_config_dict[attr])

    #     # If any `output_...` flag is set to `True`, we ensure `return_dict_in_generate` is set to `True`.
    #     if generation_config.return_dict_in_generate is False:
    #         if any(
    #             getattr(generation_config, extra_output_flag, False)
    #             for extra_output_flag in generation_config.extra_output_flags
    #         ):
    #             generation_config.return_dict_in_generate = True

    #     # Hash to detect whether the instance was modified
    #     generation_config._original_object_hash = hash(generation_config)
    #     return generation_config
