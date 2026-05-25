from dataclasses import dataclass
import logging


@dataclass
class ExperimentConfig:
    home_dir = "./features"
    seed: int = 1
    device: str = "cpu"

    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    min_lr: float = 1e-5


    batch_size: int = 32
    samples_per_lang_per_speaker = 1
    max_epochs: int = 300
    num_workers = 0

    alpha: float = 0.00
    embedding_dim: int = 512

    model_type: str = "multibranch"
    fusion: str = "linear"

    metric_loss_weight: float = 0.2
    triplet_margin: float = 0.2



    loss_face = 0.2
    loss_audio = 1.5
    loss_fusion = 1.2

    version: str = "v1"
    seen_lang: str = "English"

    audio_encoder: str = "ecappa_feats_path"


    test_missing_modality: str = "face"
    test_alpha: float = 0.0

    debug: bool = False
    log_level = logging.DEBUG if debug else logging.INFO

    early_stop: bool = True
    early_stop_patience: int = 30
    early_stop_min_delta: float = 0.0
    early_stop_metric: str = "unseen"

    @property
    def resolved_num_classes(self):
        if self.version == "v1":
            return 70
        elif self.version == "v2":
            return 84
        elif self.version == "v3":
            return 36
        else:
            raise ValueError(f"Unknown version '{self.version}'")

    @property
    def unseen_lang(self):
        mapping = {
            ("v1", "English"): "Urdu",
            ("v1", "Urdu"): "English",
            ("v2", "English"): "Hindi",
            ("v2", "Hindi"): "English",
            ("v3", "English"): "German",
            ("v3", "German"): "English",
        }
        key = (self.version, self.seen_lang)
        if key not in mapping:
            raise ValueError(
                f"Invalid version '{self.version}' or seen_lang '{self.seen_lang}'."
            )
        return mapping[key]
