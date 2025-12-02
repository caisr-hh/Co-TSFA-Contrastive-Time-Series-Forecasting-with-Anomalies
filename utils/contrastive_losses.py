import torch
import torch.nn.functional as F
import random

def Co_TSFA_contrastive_loss(orig_clean, orig_aug_list, z_clean, z_aug_list):
    """
    z_clean: [B, T, C] - clean embeddings
    z_aug_list: list of 5 tensors, each [B, T, C] - augmented embeddings
    """
    B, T, C = z_clean.shape
    if B == 1:
        return z_clean.new_tensor(0.)

    orig_dists = []
    z_dists= []
    #compute distance between original sequences with instance loss
    for orig_aug in orig_aug_list:
        # Compute instance contrastive loss for this pair
        z = torch.cat([orig_clean, orig_aug], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z_clean.device)
        orig_dist = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        orig_dists.append(orig_dist)
    #compute distances between embedded sequences with instance loss
    for z_aug in z_aug_list:

        # Compute instance contrastive loss for this pair
        z = torch.cat([z_clean, z_aug], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z_clean.device)
        z_dist = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        z_dists.append(z_dist)

    # Stack and weight losses
    orig_dists = torch.stack(orig_dists)  # [5]
    z_dists = torch.stack(z_dists)
    
    # Normalize both using log-softmax
    orig_dists_soft = F.log_softmax(orig_dists, dim=0)
    z_dists_soft = F.log_softmax(z_dists, dim=0)

    # MAE between the log-softmax vectors
    final_loss = F.l1_loss(orig_dists_soft, z_dists_soft)
    
    return final_loss

