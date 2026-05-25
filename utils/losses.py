import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjectionLoss(nn.Module):
    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        labels = labels.unsqueeze(1)

        mask = labels.eq(labels.T)
        eye = torch.eye(len(labels), device=labels.device).bool()

        pos = (mask & ~eye).float()
        neg = (~mask).float()

        dot = feats @ feats.T

        pos_mean = (pos * dot).sum() / (pos.sum() + 1e-6)
        neg_mean = (neg * dot).abs().sum() / (neg.sum() + 1e-6)

        loss = (1 - pos_mean) + 0.7 * neg_mean
        return loss


def pairwise_distance_torch(embeddings, device):
    """
    Pairwise squared Euclidean distance matrix with numerical stability.
    output[i, j] = ||x_i - x_j||^2
    """
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(dim=-1, keepdim=True)       # (B,1)
    c2 = torch.pow(precise_embeddings, 2).sum(dim=-1, keepdim=True).T     # (1,B)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)           # (B,B)

    pairwise_distances_squared = c1 + c2 - 2.0 * c3
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # explicitly zero diagonal
    batch_size = embeddings.size(0)
    eye = torch.eye(batch_size, device=device, dtype=torch.bool)
    pairwise_distances_squared = pairwise_distances_squared.masked_fill(eye, 0.0)

    return pairwise_distances_squared


class CrossLingualSemiHardTripletLoss(nn.Module):
    """
    Cross-lingual semi-hard triplet loss.

    Positive:
        same speaker, different language

    Negative:
        different speaker

    Semi-hard negative:
        negative farther than the positive distance if available;
        otherwise fallback to the hardest available negative.
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, feats, labels, lang_ids):
        device = feats.device
        feats = F.normalize(feats, dim=1)

        # squared Euclidean distance on normalized embeddings
        pdist_matrix = pairwise_distance_torch(feats, device)

        labels = labels.view(-1, 1)
        lang_ids = lang_ids.view(-1, 1)
        batch_size = labels.size(0)

        eye = torch.eye(batch_size, device=device, dtype=torch.bool)

        # cross-lingual positives: same speaker, different language, not self
        same_label = labels.eq(labels.T)
        diff_lang = ~lang_ids.eq(lang_ids.T)
        positive_mask = same_label & diff_lang & ~eye

        # negatives: different speaker
        negative_mask = ~same_label

        losses = []

        for i in range(batch_size):
            pos_idx = positive_mask[i].nonzero(as_tuple=False).squeeze(1)
            neg_idx = negative_mask[i].nonzero(as_tuple=False).squeeze(1)

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            neg_dists = pdist_matrix[i, neg_idx]

            for p in pos_idx:
                d_ap = pdist_matrix[i, p]

                # semi-hard negatives: d_an > d_ap
                semi_hard_mask = neg_dists > d_ap
                semi_hard_negatives = neg_dists[semi_hard_mask]

                if semi_hard_negatives.numel() > 0:
                    d_an = semi_hard_negatives.min()
                else:
                    # fallback: hardest available negative (closest negative)
                    d_an = neg_dists.min()

                loss_ij = F.relu(d_ap - d_an + self.margin)
                losses.append(loss_ij)

        if len(losses) == 0:
            return feats.new_tensor(0.0)

        return torch.stack(losses).mean()