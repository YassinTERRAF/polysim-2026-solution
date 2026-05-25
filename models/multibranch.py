import torch
import torch.nn as nn

from .model import (
    EmbedBranch,
    LinearFusion,
    GatedFusion,
)


class MultiBranchFOP(nn.Module):
    def __init__(self, config, face_dim, audio_dim):
        super().__init__()

        self.config = config
        emb = config.embedding_dim
        num_classes = config.resolved_num_classes

        # fusion-only projection branches
        self.face_branch = EmbedBranch(face_dim, emb)
        self.audio_branch = EmbedBranch(audio_dim, emb)

        # unimodal classifiers on raw extracted embeddings
        self.face_classifier = nn.Linear(face_dim, num_classes)
        self.audio_classifier = nn.Linear(audio_dim, num_classes)

        if config.fusion == "linear":
            self.fusion = LinearFusion(emb_dim=emb, bottleneck=256)
            fusion_dim = emb

        elif config.fusion == "gated":
            self.fusion = GatedFusion(emb)
            fusion_dim = emb

        elif config.fusion == "concat":
            self.fusion = None
            fusion_dim = emb * 2

        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        self.fusion_classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, face, audio):
        face_mask = (face.abs().sum(dim=1, keepdim=True) > 0).float()

        # raw unimodal logits
        face_logits = self.face_classifier(face)
        audio_logits = self.audio_classifier(audio)

        # fusion embeddings
        face_e = self.face_branch(face)
        audio_e = self.audio_branch(audio)

        if self.fusion is None:
            fused = torch.cat([face_e, audio_e], dim=1)
        else:
            fused, _, _ = self.fusion(face_e, audio_e, face_mask=face_mask)

        fusion_logits = self.fusion_classifier(fused)

        return {
            "face_logits": face_logits,
            "audio_logits": audio_logits,
            "fusion_logits": fusion_logits,
            "face_embed": face_e,
            "audio_embed": audio_e,
            "fusion_embed": fused,
        }