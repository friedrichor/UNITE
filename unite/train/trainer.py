import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import Trainer

from unite.model.criterions import get_sim, get_mask, get_modal_mask
from unite.utils import rank0_print, rank0_print_green



class UniteTrainer(Trainer):
    def __init__(self, *args, temp: float, has_negative: bool, dual_loss: bool, target_modal_mask: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp
        self.has_negative = has_negative
        self.dual_loss = dual_loss if not has_negative else False
        self.target_modal_mask = target_modal_mask

    def _get_model_attr(self, model, attr_name, default=None):
        """Helper method to get model attributes"""
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
        elif hasattr(model, 'module'):
            return getattr(model.module, attr_name, default)
        return default

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        ## q: query, c: candidate
        q_inputs, c_inputs, idx, target_modal = inputs['query_inputs'], inputs['candidate_inputs'], inputs['idx'], inputs['target_modal']

        q_embeds = model(**q_inputs)
        c_embeds = model(**c_inputs)
        if self.has_negative:
            neg_inputs = inputs['negative_inputs']
            neg_embeds = model(**neg_inputs)
        
        if dist.is_initialized():
            all_q_embeds = [torch.zeros_like(q_embeds) for _ in range(dist.get_world_size())]
            all_c_embeds = [torch.zeros_like(c_embeds) for _ in range(dist.get_world_size())]
            
            dist.all_gather(all_q_embeds, q_embeds)
            dist.all_gather(all_c_embeds, c_embeds)

            # # Overwrite with embeddings produced on this replace, which have the gradient.
            all_q_embeds[dist.get_rank()] = q_embeds
            all_c_embeds[dist.get_rank()] = c_embeds
            
            q_embeds = torch.cat(all_q_embeds)
            c_embeds = torch.cat(all_c_embeds)

            if idx is not None:
                all_idx = [torch.zeros_like(idx) for _ in range(dist.get_world_size())]
                dist.all_gather(all_idx, idx)
                all_idx[dist.get_rank()] = idx
                idx = torch.cat(all_idx)
            
            if self.target_modal_mask:
                all_target_modal = [torch.zeros_like(target_modal) for _ in range(dist.get_world_size())]
                dist.all_gather(all_target_modal, target_modal)
                all_target_modal[dist.get_rank()] = target_modal
                target_modal = torch.cat(all_target_modal)

            if self.has_negative:
                all_neg_embeds = [torch.zeros_like(neg_embeds) for _ in range(dist.get_world_size())]
                dist.all_gather(all_neg_embeds, neg_embeds)
                all_neg_embeds[dist.get_rank()] = neg_embeds
                neg_embeds = torch.cat(all_neg_embeds)

        rank0_print_green(f"temp: {self.temp}. has_negative: {self.has_negative}. "
                          f"dual_loss: {self.dual_loss}, target_modal_mask: {self.target_modal_mask}, "
                          f"cand_input_len: {inputs['candidate_inputs']['input_ids'].shape[1]}")

        sim_q2c, sim_c2q = get_sim(q_embeds, c_embeds, temp=self.temp)  # TODO: temp is fixed

        if self.target_modal_mask:
            modal_mask = get_modal_mask(target_modal).to(sim_q2c.device)
            sim_q2c = sim_q2c * modal_mask + (-1e9) * (1 - modal_mask)

        if self.has_negative:
            sim_q2neg, _ = get_sim(q_embeds, neg_embeds, temp=self.temp)
            sim_stacked = torch.cat([sim_q2c, sim_q2neg], 1)  # [bs, 2*bs]
            sim_targets = get_mask(sim_stacked, has_negative=True)  # [bs, 2*bs]
            loss_vtc = -torch.sum(F.log_softmax(sim_stacked, dim=1) * sim_targets, dim=1).mean()
        else:
            sim_q2c_targets = get_mask(sim_q2c, idx=idx, normalize=False)  # TODO: label, 1<->1: diagonal matrix. 1<->k: k ones at matching positions (normalize=True) or 1/k (normalize=False)
            if self.dual_loss:
                sim_c2q_targets = sim_q2c_targets
            
            loss_q2c = -torch.sum(F.log_softmax(sim_q2c, dim=1) * sim_q2c_targets, dim=1).mean()
            if self.dual_loss:
                loss_c2q = -torch.sum(F.log_softmax(sim_c2q, dim=1) * sim_c2q_targets, dim=1).mean()
                loss_vtc = (loss_q2c + loss_c2q) / 2
            else:
                loss_vtc = loss_q2c

        return loss_vtc
    
    def __save_forced_checkpoint(self):
        if self.state.global_step % self.args.force_save_steps == 0:
            # Create dedicated directory
            output_dir = os.path.join(self.args.force_save_dir, f"step_{self.state.global_step}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save checkpoint
            self.save_model(output_dir)
            print(f"Forced checkpoint saved to {output_dir}")

            # Record saving information
            with open(os.path.join(output_dir, "README.md"), "w") as f:
                f.write(f"# Forced Checkpoint\n\nStep: {self.state.global_step}")
    
    def training_step(self, *args, **kwargs):
        # Execute original training step
        output = super().training_step(*args, **kwargs)
        
        if self.args.force_save_steps > 0:
            # Check if saving is needed at each step
            if self.state.global_step % self.args.force_save_steps == 0 and self.state.global_step > 0:
                if self.is_world_process_zero():
                    self.__save_forced_checkpoint()
        
        return output
