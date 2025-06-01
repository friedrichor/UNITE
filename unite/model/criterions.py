import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def get_sim(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    temp=1.0,
    normalize=False,
):
    """calculate pair-wise video-text similarity.

    Args:
        query_embeds (torch.Tensor): The query representation. Shape: [B,D].
        candidate_embeds (torch.Tensor): The candidate representation. Shape: [B,D].
        temp (torch.Tensor): The temperature. Shape: [].
        normalize (bool): Whether to normalize the inputs.

    Returns: The similarity between query and candidate features. Shape: [B,B].
    """
    if normalize:
        # L2 normalize
        query_embeds = F.normalize(query_embeds, dim=-1)
        candidate_embeds = F.normalize(candidate_embeds, dim=-1)
    
    # Calculate similarity
    sim_q2c = torch.matmul(query_embeds, candidate_embeds.T) / temp  # (B,B)
    sim_c2q = sim_q2c.T

    return sim_q2c, sim_c2q


@torch.no_grad()
def get_mask(sim, idx=None, normalize=False, has_negative=False):
    """
    Args:
        sim (torch.Tensor): The similarity between queries and candidates. shape: (B, B).
        idx (torch.Tensor): The index for each query. Shape: [B].
        normalize (bool): If true, make row sum equal to 1
    """
    if idx is not None and not has_negative:
        # Convert idx to column vector
        idx = idx.view(-1, 1)  # [B, 1]
        # Create matching matrix by comparison
        mask = torch.eq(idx, idx.T).to(sim.dtype)  # [B, B]
        if normalize:
            # Make row sum equal to 1
            mask = mask / mask.sum(1, keepdim=True)
    else:
        # If idx is not provided or has_negative is True, create a matrix with diagonal elements set to 1
        mask = torch.zeros_like(sim)
        mask.fill_diagonal_(1)
    return mask  # `1` mark valid/matched location


def get_modal_mask(target_modals: torch.Tensor):
    """Generate a mask matrix based on target modals using bitwise operations for modal matching

    Args:
        target_modals: torch.Tensor of shape [B], each element is a bitwise encoding of modalities
                        e.g.: [1(TEXT), 2(IMAGE), 3(TEXT|IMAGE), 4(VIDEO)]
    Returns:
        mask: torch.Tensor, shape [B, B], 1 indicates modal match, 0 indicates modal mismatch
    """
    batch_size = target_modals.size(0)
    # Reshape target_modals to [B, 1] and [1, B] for broadcasting comparison
    modals_i = target_modals.view(-1, 1)  # [B, 1]
    modals_j = target_modals.view(1, -1)  # [1, B]
    
    # Set mask to 1 when two modalities are exactly the same
    mask = (modals_i == modals_j).float()
    
    return mask

