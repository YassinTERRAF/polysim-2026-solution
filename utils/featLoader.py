import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class LoadData(Dataset):
    """
    Audiovisual dataset (fully in-memory).
    """

    def __init__(
        self,
        csv_path: str,
        config,
        audio_encoder: str,
        modality: str = "audiovisual",
    ):
        assert modality == "audiovisual", (
            "This loader supports audiovisual data only."
        )

        self.audio_encoder = audio_encoder
        self.modality = modality

        df = pd.read_csv(csv_path)
        self.num_samples = len(df)

        audio_paths = [
            str((Path(config.home_dir) / p).resolve())
            for p in df[audio_encoder]
        ]
        face_paths = [
            str((Path(config.home_dir) / p).resolve())
            for p in df["facenet_feats_path"]
        ]
        labels = df["label"].astype(int).to_numpy()

        audio_feats = []
        face_feats = []

        for i in range(self.num_samples):
            audio_feats.append(np.load(audio_paths[i]).astype("float32"))
            face_feats.append(np.load(face_paths[i]).astype("float32"))

        self.audio_feats = np.stack(audio_feats)
        self.face_feats = np.stack(face_feats)
        self.labels = labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.audio_feats[idx],
            self.face_feats[idx],
            self.labels[idx],
        )


class CrossLingualTrainDataset(Dataset):
    """
    Joint train dataset over seen_lang + unseen_lang train splits.

    Returns:
        audio_feat, face_feat, label, lang_id
    """

    def __init__(self, seen_csv_path, unseen_csv_path, config, audio_encoder):
        self.audio_encoder = audio_encoder

        seen_df = pd.read_csv(seen_csv_path).copy()
        unseen_df = pd.read_csv(unseen_csv_path).copy()

        seen_df["lang_id"] = 0
        unseen_df["lang_id"] = 1

        df = pd.concat([seen_df, unseen_df], axis=0).reset_index(drop=True)
        self.num_samples = len(df)

        audio_paths = [
            str((Path(config.home_dir) / p).resolve())
            for p in df[audio_encoder]
        ]
        face_paths = [
            str((Path(config.home_dir) / p).resolve())
            for p in df["facenet_feats_path"]
        ]

        labels = df["label"].astype(int).to_numpy()
        lang_ids = df["lang_id"].astype(int).to_numpy()

        audio_feats = []
        face_feats = []

        for i in range(self.num_samples):
            audio_feats.append(np.load(audio_paths[i]).astype("float32"))
            face_feats.append(np.load(face_paths[i]).astype("float32"))

        self.audio_feats = np.stack(audio_feats)
        self.face_feats = np.stack(face_feats)
        self.labels = labels
        self.lang_ids = lang_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.audio_feats[idx],
            self.face_feats[idx],
            self.labels[idx],
            self.lang_ids[idx],
        )