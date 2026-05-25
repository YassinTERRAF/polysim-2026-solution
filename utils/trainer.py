import torch
from tqdm import tqdm
from .losses import OrthogonalProjectionLoss, CrossLingualSemiHardTripletLoss

class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config

        self.ce = torch.nn.CrossEntropyLoss()
        self.opl = OrthogonalProjectionLoss()
        self.triplet = CrossLingualSemiHardTripletLoss(
            margin=config.triplet_margin
        )
        # -----------------------------
        # Optimizer
        # -----------------------------
        if config.optimizer.lower() == "adam":
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr,
            )

        elif config.optimizer.lower() == "adamw":
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

        elif config.optimizer.lower() == "sgd":
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=0.9,
                weight_decay=config.weight_decay,
                nesterov=True,
            )

        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # -----------------------------
        # Scheduler
        # -----------------------------
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=config.max_epochs,
            eta_min=config.min_lr,
        )
    def train_epoch(self, loader, alpha, logger=None, epoch=None):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}",
            disable=not self.config.debug,
            leave=False,
        )

        for batch in pbar:
            if len(batch) == 4:
                audio, face, labels, lang_ids = batch
                lang_ids = lang_ids.to(self.config.device, non_blocking=True)
            else:
                audio, face, labels = batch
                lang_ids = None

            audio = audio.to(self.config.device, non_blocking=True)
            face = face.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            out = self.model(face, audio)

            if isinstance(out, dict):
                loss_face = self.ce(out["face_logits"], labels)
                loss_audio = self.ce(out["audio_logits"], labels)
                loss_fusion = self.ce(out["fusion_logits"], labels)

                loss = (
                    self.config.loss_face * loss_face
                    + self.config.loss_audio * loss_audio
                    + self.config.loss_fusion * loss_fusion
                )

                if alpha > 0:
                    loss = loss + alpha * self.opl(out["fusion_embed"], labels)

                if lang_ids is not None and self.config.metric_loss_weight > 0:
                    metric_loss = self.triplet(out["audio_embed"], labels, lang_ids)
                    loss = loss + self.config.metric_loss_weight * metric_loss

            else:
                fused, logits, _, _ = out
                loss = self.ce(logits, labels)

                if alpha > 0:
                    loss = loss + alpha * self.opl(fused, labels)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

        # step scheduler once per epoch
        self.scheduler.step()

        if logger is not None:
            current_lr = self.opt.param_groups[0]["lr"]
            logger.info("Epoch %03d | LR %.8f", epoch, current_lr)

        return total_loss / len(loader)