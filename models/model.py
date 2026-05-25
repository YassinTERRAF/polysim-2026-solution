import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Utility blocks
# --------------------------------------------------

def fc_block(in_dim, out_dim, p=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
    )



class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super().__init__()
        self.fc = fc_block(feat_dim, emb_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)


# --------------------------------------------------
# Linear fusion
# --------------------------------------------------

# class LinearFusion(nn.Module):
#     """
#     Learnable weighted sum fusion
#     """
#     def __init__(self):
#         super().__init__()
#         self.w_face = nn.Parameter(torch.rand(1))
#         self.w_audio = nn.Parameter(torch.rand(1))

#     def forward(self, face, audio, face_mask=None):
#         if face_mask is None:
#             face_mask = torch.ones(
#                 face.size(0), 1, device=face.device, dtype=face.dtype
#             )

#         fused = (face_mask * self.w_face) * face + self.w_audio * audio
#         return fused, face, audio


class SEVector(nn.Module):
    """
    Channel-wise attention for vector embeddings (B, D).
    """
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.fc1 = nn.Linear(channels, bottleneck)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(bottleneck, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.fc2(self.relu(self.fc1(x)))
        attn = self.sigmoid(attn)
        return x * attn



class LinearFusion(nn.Module):
    def __init__(self, emb_dim, bottleneck=256):
        super().__init__()
        self.face_se = SEVector(emb_dim, bottleneck=bottleneck)
        self.audio_se = SEVector(emb_dim, bottleneck=bottleneck)

        self.w_face = nn.Parameter(torch.rand(1))
        self.w_audio = nn.Parameter(torch.rand(1))

    def forward(self, face, audio, face_mask=None):
        if face_mask is None:
            face_mask = torch.ones(
                face.size(0), 1, device=face.device, dtype=face.dtype
            )

        # mask face before attention
        face_in = face * face_mask

        face_att = self.face_se(face_in)
        audio_att = self.audio_se(audio)

        # IMPORTANT: remove fake face signal after SE
        face_att = face_att * face_mask

        # normalize safely
        face_att = F.normalize(face_att, dim=1)
        audio_att = F.normalize(audio_att, dim=1)

        # IMPORTANT: remove fake face signal again after normalization
        face_att = face_att * face_mask

        fused = face_mask * self.w_face * face_att + self.w_audio * audio_att

        return fused, face_att, audio_att


# --------------------------------------------------
# Gated fusion
# --------------------------------------------------

class ForwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.block(x)


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion
    """
    def __init__(self, emb_dim, mid_dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            ForwardBlock(emb_dim * 2, mid_dim),
            nn.Linear(mid_dim, emb_dim),
        )

        self.face_proj = nn.Linear(emb_dim, emb_dim)
        self.audio_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, face, audio, face_mask=None):
        if face_mask is None:
            face_mask = torch.ones(
                face.size(0), 1, device=face.device, dtype=face.dtype
            )

        face_in = face * face_mask

        concat = torch.cat([face_in, audio], dim=1)
        gate = torch.sigmoid(self.attention(concat))

        face_t = torch.tanh(self.face_proj(face_in))
        audio_t = torch.tanh(self.audio_proj(audio))

        gate = gate * face_mask

        fused = gate * face_t + (1.0 - gate) * audio_t
        return fused, face_t, audio_t