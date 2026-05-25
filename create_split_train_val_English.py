#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = Path(
    "./csv_files/comp/v1_train_English.csv"
)

TRAIN_OUT = Path(
    "./csv_files/comp/v1_train_English_grouped_trainsplit_vF.csv"
)

VAL_OUT = Path(
    "./csv_files/comp/v1_train_English_grouped_valsplit_vF.csv"
)

TRAIN_RATIO = 0.8
SEED = 42


# =========================================================
# HELPERS
# =========================================================
def extract_video_id(audio_path: str) -> str:
    # example:
    # ./v1/voices/id0001/English/U3rWfLEkFvg/00000.wav
    # -> U3rWfLEkFvg
    parts = Path(audio_path).parts
    if len(parts) < 2:
        return "unknown_video"
    return parts[-2]


def grouped_split_per_identity(df: pd.DataFrame,
                               train_ratio: float = 0.8,
                               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)

    train_parts = []
    val_parts = []

    df = df.copy().reset_index(drop=True)
    df["video_id"] = df["audio_path"].apply(extract_video_id)

    for identity, sub_df in df.groupby("identity", sort=True):
        groups = sub_df["video_id"].drop_duplicates().tolist()
        rng.shuffle(groups)

        n_groups = len(groups)

        if n_groups == 1:
            # fallback: split rows if only one video group
            idx = sub_df.index.to_numpy().copy()
            rng.shuffle(idx)

            if len(idx) == 1:
                train_idx = idx
                val_idx = np.array([], dtype=int)
            else:
                n_train = max(1, int(round(train_ratio * len(idx))))
                n_train = min(n_train, len(idx) - 1)
                train_idx = idx[:n_train]
                val_idx = idx[n_train:]

            train_parts.append(df.loc[train_idx])
            if len(val_idx) > 0:
                val_parts.append(df.loc[val_idx])

        else:
            n_train_groups = max(1, int(round(train_ratio * n_groups)))
            n_train_groups = min(n_train_groups, n_groups - 1)

            train_groups = set(groups[:n_train_groups])
            val_groups = set(groups[n_train_groups:])

            train_sub = sub_df[sub_df["video_id"].isin(train_groups)]
            val_sub = sub_df[sub_df["video_id"].isin(val_groups)]

            train_parts.append(train_sub)
            val_parts.append(val_sub)

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = train_df.drop(columns=["video_id"])
    val_df = val_df.drop(columns=["video_id"])

    return train_df, val_df


def summarize(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("===== SPLIT SUMMARY =====")
    print("Train rows:", len(train_df))
    print("Val rows  :", len(val_df))
    print("Train ids :", train_df["identity"].nunique())
    print("Val ids   :", val_df["identity"].nunique())

    train_ids = set(train_df["identity"].unique())
    val_ids = set(val_df["identity"].unique())

    print("IDs missing in val  :", len(train_ids - val_ids))
    print("IDs missing in train:", len(val_ids - train_ids))

    per_id_train = train_df.groupby("identity").size()
    per_id_val = val_df.groupby("identity").size()

    print("Min train samples per id:", int(per_id_train.min()))
    print("Min val samples per id  :", int(per_id_val.min()))


# =========================================================
# MAIN
# =========================================================
def main():
    df = pd.read_csv(INPUT_CSV)

    required = {"audio_path", "face_path", "identity", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    train_df, val_df = grouped_split_per_identity(
        df,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )

    summarize(train_df, val_df)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    print("\nSaved:")
    print(TRAIN_OUT)
    print(VAL_OUT)

    print("\nUse in main.py:")
    print('train_csv = "./csv_files/comp/v1_train_English_grouped_trainsplit.csv"')
    print('test_csv = "./csv_files/comp/v1_train_English_grouped_valsplit.csv"')
    print('unseen_csv = "./csv_files/comp/v1_train_Urdu_grouped_valsplit.csv"  # create Urdu split the same way')


if __name__ == "__main__":
    main()