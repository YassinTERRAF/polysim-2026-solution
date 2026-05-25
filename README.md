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
- deterministic  refinement.

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

