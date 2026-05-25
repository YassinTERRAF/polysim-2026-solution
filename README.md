# POLYSIM 2026 Challenge Solution

This repository contains our solution for the **POLYSIM 2026 Grand Challenge** on polyglot speaker identification with missing modality.

The system addresses the four official evaluation settings:

- **P3**: In-language multimodal speaker identification
- **P4**: In-language audio-only speaker identification
- **P5**: Cross-lingual multimodal speaker identification
- **P6**: Cross-lingual audio-only speaker identification

---

## Overview

Our system is based on a multi-branch multimodal speaker identification model using audio and face representations.

The model combines:

- pretrained audio speaker embeddings,
- pretrained face embeddings,
- unimodal classification heads,
- multimodal fusion,
- metric-learning regularization,
- deterministic refinement.

---

## Feature Extraction

### Audio Embeddings

Audio embeddings are extracted using a **ResNet293_LM** model pretrained on **VoxCeleb1** and **VoxCeleb2**.

### Face Embeddings

Face embeddings are extracted using **FaceNet**, with face detection performed using **MTCNN**.

The extracted features are used as fixed input representations for the multimodal classification model.

---

## Model Architecture

The model follows a multi-branch design composed of:

- an audio branch,
- a face branch,
- unimodal classifiers,
- a multimodal fusion branch.

The audio and face features are first projected into a shared embedding space. The fusion module combines the two modalities using a learnable attention-based fusion mechanism.

The model outputs:

- audio logits,
- face logits,
- fusion logits,
- audio embeddings,
- face embeddings,
- fusion embeddings.

This design allows the same model to operate under both complete-modality and missing-modality inference settings.

---

## Training Objective

The training objective combines:

- cross-entropy loss for face classification,
- cross-entropy loss for audio classification,
- cross-entropy loss for fusion classification,
- orthogonal projection loss on fusion embeddings,
- cross-lingual semi-hard triplet loss on audio embeddings.

The triplet loss encourages speaker identity consistency across languages by pulling together samples from the same speaker across different languages while separating different speakers.

---

## Running the Code

### Training

To train the multimodal model:

```bash
python main.py
```

Training configuration and hyperparameters are defined in:

```text
config.py
```

---

### Submission Generation

To generate the official POLYSIM submission CSV files:

```bash
python submit.py
```

The generated submission files are saved under:

```text
csv_files/submission/
```

The script generates:

- `submission_v1_test_English_English.csv`
- `submission_v1_test_English_Urdu.csv`

corresponding to the official POLYSIM evaluation settings.

---

## Repository Structure

```text
models/
    model.py
    multibranch.py

utils/
    prediction_refinement.py
    refinement_config.py
    losses.py
    trainer.py

submit.py
main.py
config.py
```

---

## Notes

Large artifacts are not included in this repository, including:

- extracted feature files,
- pretrained checkpoints,
- challenge datasets,
- generated submissions.

These files should be placed locally according to the paths defined in the configuration files.

---

## Dependencies

Main dependencies include:

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Acknowledgment

This repository was developed for participation in the **POLYSIM 2026 Grand Challenge**.

We thank the POLYSIM organizing team for preparing the benchmark and evaluation protocol.
