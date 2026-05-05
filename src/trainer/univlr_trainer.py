import os
import torch
import torch.nn as nn
import wandb
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenUniVLRSFTTrainer(Trainer):

    def __init__(self, *args, temp_folder=None, oci_handler=None, **kwargs):
        super(QwenUniVLRSFTTrainer, self).__init__(*args, **kwargs)
        # if online checkpointing
        
        self.oci_handler = oci_handler
        self.temp_folder = temp_folder     # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"

    @staticmethod
    def _distributed_barrier():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            univlr_head_parameters =[]

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
            if self.args.univlr_head_lr is not None:
                lr_mapper["univlr_head"] = self.args.univlr_head_lr
                univlr_head_parameters = [name for name, _ in opt_model.named_parameters() if "univlr_head" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters + univlr_head_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
                
                if univlr_head_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in univlr_head_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.univlr_head_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in univlr_head_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.univlr_head_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        # modified to support online checkpointing
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        # output_dir is the local path forcheckpoint
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # output_dir is local; now we save to cloud if needed
        if self.temp_folder:
            self._distributed_barrier()
        if self.temp_folder:
            # In ZeRO/DeepSpeed runs, each rank may own distinct optimizer shards locally.
            remote_chkpt_folder = os.path.join(self.args.remote_output_dir,checkpoint_folder)
            if remote_chkpt_folder[0] == '/':
                remote_chkpt_folder = remote_chkpt_folder[1:]       #remote pathing rules will take bucket//checkpoints, need to remove the dup
            self.oci_handler.save_checkpoint(output_dir,remote_chkpt_folder)    #save local chkpt to remote folder
            # remove the local 
            self.temp_folder.cleanup(checkpoint_name=checkpoint_folder)
        if self.temp_folder:
            self._distributed_barrier()


        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def compute_loss(self, model, inputs,num_items_in_batch=None, return_outputs=False):

        if self.args.enable_data_packing:
            batch_size = inputs['input_ids'].size(0)
            total_tokens = inputs['input_ids'].size(0) * inputs['input_ids'].size(1)
            self.log({
            "batch_size": batch_size,
            "tokens_per_device": total_tokens,})

        outputs = model(**inputs)
        # loss = outputs.loss  # total loss
        loss_ce = outputs.loss_ce
        loss_univlr = outputs.loss_univlr
        loss_mode_switch = outputs.loss_mode_switch
        loss_text_latent = getattr(outputs, "loss_text_latent", None)
        loss_image_latent = getattr(outputs, "loss_image_latent", None)
        loss_latent_nll = getattr(outputs, "loss_latent_nll", None)
        loss_latent_kl = getattr(outputs, "loss_latent_kl", None)
        loss_latent_det = getattr(outputs, "loss_latent_det", None)
        loss_latent_logvar_reg = getattr(outputs, "loss_latent_logvar_reg", None)
        latent_logvar_q_mean = getattr(outputs, "latent_logvar_q_mean", None)
        latent_logvar_p_mean = getattr(outputs, "latent_logvar_p_mean", None)
        univlr_replay_ratio = getattr(outputs, "univlr_replay_ratio", None)
        loss_univlr_lambda = float(getattr(self.args, "loss_univlr_lambda", 1.0))
        loss_vae_nll_weight = float(getattr(self.args, "loss_vae_nll_weight", 1.0))
        loss_vae_kl_beta = float(getattr(self.args, "loss_vae_kl_beta", 1e-3))
        loss_vae_det_weight = float(getattr(self.args, "loss_vae_det_weight", 0.0))
        loss_vae_logvar_reg_weight = float(getattr(self.args, "loss_vae_logvar_reg_weight", 0.0))

        if self.args.mode_switch_loss:
            loss = loss_ce + self.args.loss_univlr_lambda * loss_univlr + self.args.loss_mode_switch_lambda * loss_mode_switch
        else:
            loss = loss_ce + self.args.loss_univlr_lambda * loss_univlr if self.args.loss_univlr_lambda > 0 else loss_ce

        # Log each component
        self.log({
            "loss_total": loss.detach().item(),
            "loss_ce": loss_ce.detach().item(),
            "loss_univlr": loss_univlr.detach().item() if loss_univlr is not None else 0.0,
            "loss_univlr_scaled": (loss_univlr_lambda * loss_univlr.detach().item()) if loss_univlr is not None else 0.0,
            "loss_mode_switch": loss_mode_switch.detach().item() if loss_mode_switch is not None else 0.0,
            "loss_text_latent": loss_text_latent.detach().item() if loss_text_latent is not None else 0.0,
            "loss_image_latent": loss_image_latent.detach().item() if loss_image_latent is not None else 0.0,
            "loss_latent_nll": loss_latent_nll.detach().item() if loss_latent_nll is not None else 0.0,
            "loss_latent_kl": loss_latent_kl.detach().item() if loss_latent_kl is not None else 0.0,
            "loss_latent_nll_weighted": (loss_vae_nll_weight * loss_latent_nll.detach().item()) if loss_latent_nll is not None else 0.0,
            "loss_latent_kl_weighted": (loss_vae_kl_beta * loss_latent_kl.detach().item()) if loss_latent_kl is not None else 0.0,
            "loss_latent_det": loss_latent_det.detach().item() if loss_latent_det is not None else 0.0,
            "loss_latent_det_weighted": (loss_vae_det_weight * loss_latent_det.detach().item()) if loss_latent_det is not None else 0.0,
            "loss_latent_logvar_reg": loss_latent_logvar_reg.detach().item() if loss_latent_logvar_reg is not None else 0.0,
            "loss_latent_logvar_reg_weighted": (loss_vae_logvar_reg_weight * loss_latent_logvar_reg.detach().item()) if loss_latent_logvar_reg is not None else 0.0,
            "latent_logvar_q_mean": latent_logvar_q_mean.detach().item() if latent_logvar_q_mean is not None else 0.0,
            "latent_logvar_p_mean": latent_logvar_p_mean.detach().item() if latent_logvar_p_mean is not None else 0.0,
            "univlr_replay_ratio": univlr_replay_ratio.detach().item() if univlr_replay_ratio is not None else 0.0,
        })


        return (loss, outputs) if return_outputs else loss
