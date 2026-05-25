import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from utils.refinement_config import (
    TRAIN_FACE_ROOT,
    TRAIN_AUDIO_ROOT,
    TEST_FEATURE_ROOT,
    DEBUG_DIR,
    FIRST_PASS_MIN_SCORE_STRONG,
    FIRST_PASS_MIN_SCORE_WEAK,
    RUN_SHARED_CONSISTENCY_REFINEMENT,
    SHARED_CLUSTER_THRESHOLD,
    SHARED_MIN_FACE_GAP,
    SHARED_MIN_VIDEO_GAP,
    SHARED_MIN_COMBINED_SCORE,
    SHARED_AUDIO_BLOCK_GAP,
    SHARED_MAX_REFINEMENTS,
    RUN_LONG_TRACK_VIDEO_REFINEMENT,
    LONG_TRACK_TOPK_VIDEO,
    LONG_TRACK_MAX_CONFIDENCE,
    LONG_TRACK_MAX_FACE_RANK,
    LONG_TRACK_MIN_VIDEO_VOTES,
    LONG_TRACK_MIN_TOTAL_FRAMES,
    LONG_TRACK_MIN_FRAMES_PER_VOTE,
    LONG_TRACK_MIN_VIDEO_SCORE,
    LONG_TRACK_MIN_AUDIO_RANK,
    LONG_TRACK_MAX_REFINEMENTS,
    GROUP_CLUSTER_THRESHOLD,
    GROUP_MIN_CLUSTER_SIZE,
    GROUP_MIN_MAJORITY_RATIO,
    GROUP_SEEN_MIN_RATIO,
    GROUP_SEEN_MIN_SIZE,
    GROUP_MAX_CHANGES_SEEN,
    GROUP_MAX_CHANGES_UNSEEN,
)


def normalize(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + 1e-9)


def label_from_folder(folder):
    return int(folder.replace("id", "")) - 1


def label_to_folder(label):
    return f"id{int(label) + 1:04d}"


def fix_rel_path(p):
    p = str(p)
    if p.startswith("test/"):
        p = p[len("test/"):]
    return p


def test_face_path(face_rel):
    return os.path.join(TEST_FEATURE_ROOT, fix_rel_path(face_rel).replace(".jpg", ".npy"))


def test_audio_path(voice_rel):
    return os.path.join(TEST_FEATURE_ROOT, fix_rel_path(voice_rel).replace(".wav", ".npy"))


def build_mean_prototypes(root, lang):
    labels = []
    protos = []

    for spk_dir in sorted(glob.glob(os.path.join(root, "id*"))):
        label = label_from_folder(os.path.basename(spk_dir))
        files = sorted(glob.glob(os.path.join(spk_dir, lang, "**", "*.npy"), recursive=True))

        embs = []
        for f in files:
            try:
                embs.append(normalize(np.load(f)))
            except Exception:
                continue

        if embs:
            labels.append(label)
            protos.append(normalize(np.mean(np.stack(embs), axis=0)))

    return np.asarray(labels, dtype=np.int64), np.stack(protos).astype(np.float32)


def build_video_level_prototypes(root, lang):
    labels = []
    protos = []
    video_ids = []

    for spk_dir in sorted(glob.glob(os.path.join(root, "id*"))):
        label = label_from_folder(os.path.basename(spk_dir))
        lang_dir = os.path.join(spk_dir, lang)

        if not os.path.isdir(lang_dir):
            continue

        for video_dir in sorted(glob.glob(os.path.join(lang_dir, "*"))):
            if not os.path.isdir(video_dir):
                continue

            files = sorted(glob.glob(os.path.join(video_dir, "*.npy")))
            embs = []
            for f in files:
                try:
                    embs.append(normalize(np.load(f)))
                except Exception:
                    continue

            if embs:
                labels.append(label)
                video_ids.append(os.path.basename(video_dir))
                protos.append(normalize(np.mean(np.stack(embs), axis=0)))

    return np.asarray(labels, dtype=np.int64), np.stack(protos).astype(np.float32), np.asarray(video_ids)


def rank_proto(x, labels, protos):
    sims = protos @ x
    order = np.argsort(-sims)
    return [{"label": int(labels[idx]), "score": float(sims[idx])} for idx in order]


def rank_proto_order(x, labels, protos):
    sims = protos @ x
    return np.argsort(-sims), sims


def find_rank_score(ranked, label):
    for rank, item in enumerate(ranked, start=1):
        if int(item["label"]) == int(label):
            return rank, float(item["score"])
    return 999, -1.0


def rank_for_label(order, labels, sims, label):
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == int(label):
            return rank, float(sims[idx]), idx
    return 999999, -999.0, None


def top_label(order, labels, sims):
    idx = order[0]
    return int(labels[idx]), float(sims[idx]), idx


def rank_refinement_candidates(sub_df, test_df, lang, anchor_col, target_col):
    df = sub_df.merge(test_df[["key", "voices", "faces"]], on="key", how="left")

    face_labels, face_protos = build_mean_prototypes(TRAIN_FACE_ROOT, lang)
    audio_labels, audio_protos = build_mean_prototypes(TRAIN_AUDIO_ROOT, lang)

    rows = []

    for _, row in df.iterrows():
        key = row["key"]
        anchor = int(row[anchor_col])
        current = int(row[target_col])

        if current == anchor:
            continue

        face_emb = normalize(np.load(test_face_path(row["faces"])))
        audio_emb = normalize(np.load(test_audio_path(row["voices"])))

        face_ranked = rank_proto(face_emb, face_labels, face_protos)
        audio_ranked = rank_proto(audio_emb, audio_labels, audio_protos)

        face_top = int(face_ranked[0]["label"])
        audio_top = int(audio_ranked[0]["label"])

        anchor_face_rank, anchor_face_score = find_rank_score(face_ranked, anchor)
        current_face_rank, current_face_score = find_rank_score(face_ranked, current)
        anchor_audio_rank, anchor_audio_score = find_rank_score(audio_ranked, anchor)
        current_audio_rank, current_audio_score = find_rank_score(audio_ranked, current)

        face_gap = anchor_face_score - current_face_score
        audio_gap = anchor_audio_score - current_audio_score

        face_top_is_anchor = face_top == anchor
        audio_top_is_anchor = audio_top == anchor
        current_bad_face = current_face_rank >= 3
        current_bad_audio = current_audio_rank >= 3

        score = 0.0
        score += max(0.0, face_gap) * 10.0
        score += max(0.0, audio_gap) * 5.0
        score += int(face_top_is_anchor) * 5.0
        score += int(audio_top_is_anchor) * 3.0
        score += int(current_bad_face) * 1.5
        score += int(current_bad_audio) * 0.8
        score += min(current_face_rank, 70) * 0.03
        score += min(current_audio_rank, 70) * 0.01

        risk = []
        if face_gap < 0:
            risk.append("face_prefers_current")
        if audio_gap < -0.05:
            risk.append("audio_strongly_prefers_current")
        if not face_top_is_anchor and not audio_top_is_anchor:
            risk.append("neither_top_is_anchor")

        if len(risk) == 0:
            risk_level = "low"
        elif len(risk) == 1:
            risk_level = "medium"
        else:
            risk_level = "high"

        rows.append({
            "key": key,
            "suggested_label": anchor,
            "risk_level": risk_level,
            "score": round(score, 4),
        })

    candidates = pd.DataFrame(rows)

    if len(candidates) > 0:
        risk_order = {"low": 0, "medium": 1, "high": 2}
        candidates["risk_order"] = candidates["risk_level"].map(risk_order)
        candidates = candidates.sort_values(["risk_order", "score"], ascending=[True, False]).drop(columns=["risk_order"])

    return candidates


def apply_score_threshold_candidates(sub_df, candidates, target_col):
    refined = sub_df.copy()

    if len(candidates) == 0:
        return refined

    threshold = FIRST_PASS_MIN_SCORE_STRONG if target_col == "p4" else FIRST_PASS_MIN_SCORE_WEAK
    selected = candidates[candidates["score"] >= threshold].copy()
    selected_map = dict(zip(selected["key"], selected["suggested_label"]))

    for i in range(len(refined)):
        key = refined.loc[i, "key"]
        if key in selected_map:
            refined.loc[i, target_col] = int(selected_map[key])

    return refined


def shared_consistency_refinement(sub_df, test_df, lang):
    refined = sub_df.copy()
    df = refined.merge(test_df[["key", "voices", "faces"]], on="key", how="left").reset_index(drop=True)

    face_labels, face_protos = build_mean_prototypes(TRAIN_FACE_ROOT, lang)
    audio_labels, audio_protos = build_mean_prototypes(TRAIN_AUDIO_ROOT, lang)
    vf_labels, vf_protos, _ = build_video_level_prototypes(TRAIN_FACE_ROOT, lang)

    face_embs = []
    audio_embs = []
    for _, row in df.iterrows():
        face_embs.append(normalize(np.load(test_face_path(row["faces"]))))
        audio_embs.append(normalize(np.load(test_audio_path(row["voices"]))))

    face_embs = np.stack(face_embs).astype(np.float32)
    audio_embs = np.stack(audio_embs).astype(np.float32)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=SHARED_CLUSTER_THRESHOLD,
    )
    df["face_group"] = clustering.fit_predict(face_embs)

    group_majority = {}
    group_size = {}
    group_ratio = {}
    for gid, g in df.groupby("face_group"):
        counts = g["p3"].value_counts()
        group_majority[int(gid)] = int(counts.idxmax())
        group_size[int(gid)] = int(len(g))
        group_ratio[int(gid)] = int(counts.max()) / max(int(len(g)), 1)

    rows = []

    for i, row in df.iterrows():
        key = row["key"]
        p3 = int(row["p3"])
        p4 = int(row["p4"])
        current = p3

        face_emb = face_embs[i]
        audio_emb = audio_embs[i]

        face_order, face_sims = rank_proto_order(face_emb, face_labels, face_protos)
        audio_order, audio_sims = rank_proto_order(audio_emb, audio_labels, audio_protos)
        vf_order, vf_sims = rank_proto_order(face_emb, vf_labels, vf_protos)

        face_top, face_top_score, _ = top_label(face_order, face_labels, face_sims)
        audio_top, audio_top_score, _ = top_label(audio_order, audio_labels, audio_sims)
        vf_top, vf_top_score, _ = top_label(vf_order, vf_labels, vf_sims)

        cur_face_rank, cur_face_score, _ = rank_for_label(face_order, face_labels, face_sims, current)
        cur_audio_rank, cur_audio_score, _ = rank_for_label(audio_order, audio_labels, audio_sims, current)
        cur_vf_rank, cur_vf_score, _ = rank_for_label(vf_order, vf_labels, vf_sims, current)

        face_gap = face_top_score - cur_face_score
        audio_gap = audio_top_score - cur_audio_score
        vf_gap = vf_top_score - cur_vf_score

        gid = int(row["face_group"])
        gmaj = int(group_majority[gid])
        gsize = int(group_size[gid])
        gratio = float(group_ratio[gid])

        suggestions = []
        if face_top != current:
            suggestions.append(face_top)
        if vf_top != current:
            suggestions.append(vf_top)
        if audio_top != current:
            suggestions.append(audio_top)
        if gmaj != current and gsize >= 2:
            suggestions.append(gmaj)

        if not suggestions:
            continue

        vote_counts = pd.Series(suggestions).value_counts()
        suggested = int(vote_counts.idxmax())
        suggestion_votes = int(vote_counts.max())

        face_support = face_top == suggested and face_gap >= SHARED_MIN_FACE_GAP
        video_support = vf_top == suggested and vf_gap >= SHARED_MIN_VIDEO_GAP
        audio_support = audio_top == suggested and audio_gap >= -0.02
        group_support = gmaj == suggested and gsize >= 2 and gratio >= 0.50
        audio_blocks = audio_top == current and audio_gap <= -SHARED_AUDIO_BLOCK_GAP

        score = 0.0
        score += int(face_support) * 4.0
        score += int(video_support) * 5.0
        score += int(audio_support) * 1.5
        score += int(group_support) * 2.5
        score += max(0.0, face_gap) * 10.0
        score += max(0.0, vf_gap) * 12.0
        score += max(0.0, audio_gap) * 3.0
        score += min(cur_face_rank, 50) * 0.03
        score += min(cur_vf_rank, 50) * 0.04
        score += suggestion_votes * 0.8
        score -= int(audio_blocks) * 4.0

        accept = False
        if face_support and video_support:
            accept = True
        elif face_support and group_support:
            accept = True
        elif video_support and group_support:
            accept = True
        elif face_support and audio_support and suggestion_votes >= 2:
            accept = True
        elif video_support and audio_support and suggestion_votes >= 2:
            accept = True
        if audio_blocks:
            accept = False

        if accept and score >= SHARED_MIN_COMBINED_SCORE:
            rows.append({"key": key, "suggested_label": suggested, "score": round(score, 5)})

    candidates = pd.DataFrame(rows)
    if len(candidates) == 0:
        return refined

    candidates = candidates.sort_values("score", ascending=False).head(SHARED_MAX_REFINEMENTS)
    selected_map = dict(zip(candidates["key"], candidates["suggested_label"]))

    for i in range(len(refined)):
        key = refined.loc[i, "key"]
        if key in selected_map:
            refined.loc[i, "p3"] = int(selected_map[key])
            refined.loc[i, "p4"] = int(selected_map[key])

    return refined


def group_majority_refinement(sub_df, test_df, task_cols, lang_name):
    df = sub_df.merge(test_df[["key", "faces", "voices"]], on="key", how="left")

    embs = []
    valid_rows = []
    for _, row in df.iterrows():
        path = test_face_path(row["faces"])
        if os.path.exists(path):
            embs.append(normalize(np.load(path)))
            valid_rows.append(row)

    if not embs:
        return sub_df.copy()

    embs = np.stack(embs)
    work = pd.DataFrame(valid_rows).reset_index(drop=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=GROUP_CLUSTER_THRESHOLD,
    )
    work["face_group"] = clustering.fit_predict(embs)

    refined = sub_df.copy()

    for task_col in task_cols:
        rows = []

        for gid, g in work.groupby("face_group"):
            group_size = len(g)
            if group_size < GROUP_MIN_CLUSTER_SIZE:
                continue

            counts = g[task_col].value_counts()
            majority_label = int(counts.idxmax())
            majority_count = int(counts.max())
            majority_ratio = majority_count / group_size

            if majority_ratio < GROUP_MIN_MAJORITY_RATIO:
                continue

            for _, r in g.iterrows():
                current_label = int(r[task_col])
                if current_label != majority_label:
                    rows.append({
                        "key": r["key"],
                        "group_size": int(group_size),
                        "suggested_label": majority_label,
                        "majority_ratio": round(majority_ratio, 5),
                    })

        candidates = pd.DataFrame(rows)
        if len(candidates) == 0:
            continue

        candidates = candidates.sort_values(["majority_ratio", "group_size"], ascending=[False, False])

        if task_col in ["p3", "p4"]:
            candidates = candidates[
                (candidates["majority_ratio"] >= GROUP_SEEN_MIN_RATIO)
                & (candidates["group_size"] >= GROUP_SEEN_MIN_SIZE)
            ].copy()
            limit = GROUP_MAX_CHANGES_SEEN
        elif task_col == "p6":
            limit = GROUP_MAX_CHANGES_UNSEEN
        else:
            limit = 1

        selected = candidates.head(min(limit, len(candidates)))
        selected_map = dict(zip(selected["key"], selected["suggested_label"]))

        for i in range(len(refined)):
            key = refined.loc[i, "key"]
            if key in selected_map:
                refined.loc[i, task_col] = int(selected_map[key])

    return refined


def video_track_consistency_refinement(sub_df, test_df, lang):
    refined = sub_df.copy()
    df = refined.merge(test_df[["key", "voices", "faces"]], on="key", how="left").reset_index(drop=True)

    if "_p3_confidence" not in df.columns:
        return refined

    face_labels, face_protos = build_mean_prototypes(TRAIN_FACE_ROOT, lang)
    audio_labels, audio_protos = build_mean_prototypes(TRAIN_AUDIO_ROOT, lang)
    vf_labels, vf_protos, vf_videos = build_video_level_prototypes(TRAIN_FACE_ROOT, lang)

    rows = []

    for _, row in df.iterrows():
        key = row["key"]
        current = int(row["p3"])
        p3_conf = float(row["_p3_confidence"])

        face_emb = normalize(np.load(test_face_path(row["faces"])))
        audio_emb = normalize(np.load(test_audio_path(row["voices"])))

        face_order, face_sims = rank_proto_order(face_emb, face_labels, face_protos)
        audio_order, audio_sims = rank_proto_order(audio_emb, audio_labels, audio_protos)

        current_audio_rank, _, _ = rank_for_label(audio_order, audio_labels, audio_sims, current)

        vf_sims = vf_protos @ face_emb
        vf_order = np.argsort(-vf_sims)[:LONG_TRACK_TOPK_VIDEO]

        video_rows = []
        for idx in vf_order:
            label = int(vf_labels[idx])
            video_rows.append({
                "label": label,
                "score": float(vf_sims[idx]),
                "frames": int(len(glob.glob(os.path.join(
                    TRAIN_FACE_ROOT,
                    label_to_folder(label),
                    lang,
                    str(vf_videos[idx]),
                    "*.npy",
                )))),
            })

        video_df = pd.DataFrame(video_rows)
        if len(video_df) == 0:
            continue

        video_agg = video_df.groupby("label").agg(
            video_vote_count=("label", "size"),
            video_best_score=("score", "max"),
            video_total_frames=("frames", "sum"),
        ).reset_index()

        for _, cand in video_agg.iterrows():
            label = int(cand["label"])
            if label == current:
                continue

            person_face_rank, _, _ = rank_for_label(face_order, face_labels, face_sims, label)

            video_vote_count = int(cand["video_vote_count"])
            video_best_score = float(cand["video_best_score"])
            video_total_frames = int(cand["video_total_frames"])
            frames_per_vote = video_total_frames / max(video_vote_count, 1)

            passes = (
                p3_conf <= LONG_TRACK_MAX_CONFIDENCE
                and person_face_rank <= LONG_TRACK_MAX_FACE_RANK
                and video_vote_count >= LONG_TRACK_MIN_VIDEO_VOTES
                and video_total_frames >= LONG_TRACK_MIN_TOTAL_FRAMES
                and frames_per_vote >= LONG_TRACK_MIN_FRAMES_PER_VOTE
                and video_best_score >= LONG_TRACK_MIN_VIDEO_SCORE
                and current_audio_rank > LONG_TRACK_MIN_AUDIO_RANK
            )

            if not passes:
                continue

            score = (
                frames_per_vote * 0.15
                + video_total_frames * 0.01
                + video_best_score * 5.0
                - person_face_rank * 0.05
                + max(0.0, LONG_TRACK_MAX_CONFIDENCE - p3_conf) * 5.0
            )

            rows.append({"key": key, "suggested_label": label, "score": round(score, 5)})

    candidates = pd.DataFrame(rows)
    if len(candidates) == 0:
        return refined

    candidates = candidates.sort_values("score", ascending=False).head(LONG_TRACK_MAX_REFINEMENTS)
    selected_map = dict(zip(candidates["key"], candidates["suggested_label"]))

    for i in range(len(refined)):
        key = refined.loc[i, "key"]
        if key in selected_map:
            refined.loc[i, "p3"] = int(selected_map[key])
            refined.loc[i, "p4"] = int(selected_map[key])

    return refined


def refine_predictions(
    submission_seen_raw,
    submission_unseen_raw,
    seen_csv,
    unseen_csv,
    seen_lang,
    unseen_lang,
    debug_dir=None,
):
    submission_seen_refined = apply_score_threshold_candidates(
        sub_df=submission_seen_raw,
        candidates=rank_refinement_candidates(
            sub_df=submission_seen_raw,
            test_df=seen_csv,
            lang=seen_lang,
            anchor_col="p3",
            target_col="p4",
        ),
        target_col="p4",
    )

    submission_unseen_refined = apply_score_threshold_candidates(
        sub_df=submission_unseen_raw,
        candidates=rank_refinement_candidates(
            sub_df=submission_unseen_raw,
            test_df=unseen_csv,
            lang=unseen_lang,
            anchor_col="p5",
            target_col="p6",
        ),
        target_col="p6",
    )

    if RUN_SHARED_CONSISTENCY_REFINEMENT:
        submission_seen_refined = shared_consistency_refinement(
            sub_df=submission_seen_refined,
            test_df=seen_csv,
            lang=seen_lang,
        )

    submission_seen_refined = group_majority_refinement(
        sub_df=submission_seen_refined,
        test_df=seen_csv,
        task_cols=["p3", "p4"],
        lang_name=seen_lang,
    )

    submission_unseen_refined = group_majority_refinement(
        sub_df=submission_unseen_refined,
        test_df=unseen_csv,
        task_cols=["p5", "p6"],
        lang_name=unseen_lang,
    )

    if RUN_LONG_TRACK_VIDEO_REFINEMENT:
        submission_seen_refined = video_track_consistency_refinement(
            sub_df=submission_seen_refined,
            test_df=seen_csv,
            lang=seen_lang,
        )

    return submission_seen_refined, submission_unseen_refined
