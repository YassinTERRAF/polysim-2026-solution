import logging
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig

from utils.featLoader import LoadData, CrossLingualTrainDataset
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.earlystop import EarlyStopping

from models.fop import FOP
from models.multibranch import MultiBranchFOP
from utils.samplers import CrossLingualPairBatchSampler
import os


def save_checkpoint(model, optimizer, config, epoch, metric_value, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "metric": metric_value,
        "early_stop_metric": config.early_stop_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "config": vars(config),
    }

    torch.save(checkpoint, save_path)


def setup_logger(config):
    logger = logging.getLogger("Experiment")
    logger.setLevel(config.log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def make_eval_loader(csv_path, config, shuffle=False):
    dataset = LoadData(
        csv_path=csv_path,
        config=config,
        audio_encoder=config.audio_encoder,
        modality="audiovisual",
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def make_crosslingual_train_loader(seen_csv, unseen_csv, config):
    dataset = CrossLingualTrainDataset(
        seen_csv_path=seen_csv,
        unseen_csv_path=unseen_csv,
        config=config,
        audio_encoder=config.audio_encoder,
    )

    batch_sampler = CrossLingualPairBatchSampler(
        dataset=dataset,
        batch_size=config.batch_size,
        samples_per_lang_per_speaker=config.samples_per_lang_per_speaker,
        drop_last=False,
        seed=config.seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return dataset, loader


def main():
    config = ExperimentConfig()
    torch.manual_seed(config.seed)

    if config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger = setup_logger(config)

    logger.info("=== Experiment started ===")
    logger.info(
        "Seed=%d | Device=%s | Model=%s | Fusion=%s | Version=%s | Seen=%s | Unseen=%s | "
        "AudioEncoder=%s | Alpha=%.3f | MetricLoss=%.3f",
        config.seed,
        config.device,
        config.model_type,
        config.fusion,
        config.version,
        config.seen_lang,
        config.unseen_lang,
        config.audio_encoder,
        config.alpha,
        config.metric_loss_weight,
    )

    # --------------------------------------------------
    # CSV paths
    # --------------------------------------------------
    train_seen_csv = f"./csv_files/comp/{config.version}_train_{config.seen_lang}_grouped_trainsplit_vF.csv"
    train_unseen_csv = f"./csv_files/comp/{config.version}_train_{config.unseen_lang}_grouped_trainsplit_vF.csv"

    val_seen_csv = f"./csv_files/comp/{config.version}_train_{config.seen_lang}_grouped_valsplit_vF.csv"
    val_unseen_csv = f"./csv_files/comp/{config.version}_train_{config.unseen_lang}_grouped_valsplit_vF.csv"

    logger.info("Train seen CSV  : %s", train_seen_csv)
    logger.info("Train unseen CSV: %s", train_unseen_csv)
    logger.info("Val seen CSV    : %s", val_seen_csv)
    logger.info("Val unseen CSV  : %s", val_unseen_csv)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    _, train_loader = make_crosslingual_train_loader(
        train_seen_csv,
        train_unseen_csv,
        config,
    )

    val_seen_dataset, _ = make_eval_loader(val_seen_csv, config, shuffle=False)
    val_unseen_dataset, _ = make_eval_loader(val_unseen_csv, config, shuffle=False)

    # --------------------------------------------------
    # Infer feature dimensions
    # --------------------------------------------------
    batch = next(iter(train_loader))
    audio, face, labels, langs = batch

    logger.info("Feature dimensions | Audio=%d | Face=%d", audio.shape[1], face.shape[1])

    model_type = config.model_type.lower()

    if model_type == "fop":
        model = FOP(
            config=config,
            face_dim=face.shape[1],
            audio_dim=audio.shape[1],
        )
    elif model_type == "multibranch":
        model = MultiBranchFOP(
            config=config,
            face_dim=face.shape[1],
            audio_dim=audio.shape[1],
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    logger.info(
        "Model initialized | Params=%.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    trainer = Trainer(model, config)
    evaluator = Evaluator(model, config)

    alpha = config.alpha
    logger.info("=== Training with alpha=%.3f ===", alpha)

    best_metric = -float("inf")
    best_epoch = -1

    save_path = (
        f"./checkpoints/"
        f"{config.version}_{config.seen_lang}_"
        f"{config.audio_encoder.replace('_feats_path', '')}_"
        f"alpha{alpha}_metric{config.metric_loss_weight}_best.pt"
    )

    early_stopper = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
    )

    for epoch in range(config.max_epochs):
        loss = trainer.train_epoch(train_loader, alpha, logger=logger, epoch=epoch)

        p3_proxy = evaluator.accuracy(val_seen_dataset, head="fusion")
        p4_proxy = evaluator.accuracy_missing_face(val_seen_dataset, head="fusion")
        p5_proxy = evaluator.accuracy(val_unseen_dataset, head="fusion")
        p6_proxy = evaluator.accuracy_missing_face(val_unseen_dataset, head="fusion")

        monitor_value = (p3_proxy + p4_proxy + p5_proxy + p6_proxy) / 4.0

        if monitor_value > best_metric:
            best_metric = monitor_value
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=trainer.opt,
                config=config,
                epoch=epoch,
                metric_value=monitor_value,
                save_path=save_path,
            )

        logger.info(
            "Epoch %03d | Loss %.4f | P3 %.2f | P4 %.2f | P5 %.2f | P6 %.2f | Avg %.2f | Best %.2f @ %d",
            epoch,
            loss,
            p3_proxy,
            p4_proxy,
            p5_proxy,
            p6_proxy,
            monitor_value,
            best_metric,
            best_epoch,
        )
        if config.early_stop and early_stopper.step(monitor_value):
            logger.info(
                "Early stopping triggered at epoch %d (best %s accuracy = %.2f at epoch %d)",
                epoch,
                config.early_stop_metric,
                early_stopper.best_score,
                best_epoch,
            )
            break

    logger.info("=== Experiment finished ===")


if __name__ == "__main__":
    main()