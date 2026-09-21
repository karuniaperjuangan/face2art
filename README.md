# FFHQ to SNGFaces with EnCo

This repository is a compact PyTorch implementation of one-way unpaired image
translation from real FFHQ faces to SNGFaces paintings. It keeps the code small
enough to study while exposing datasets, architectures, loss weights, and the
training schedule through YAML.

The model follows the content objective from
[EnCo](https://github.com/XiudingCai/EnCo-pytorch): a ResNet generator is trained
with a PatchGAN discriminator, while paired encoder/decoder features are sampled
with discriminator-aware guidance and matched by stop-gradient cosine
regression. The original target-domain L1 identity branch is retained. An
additional frozen [AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) encoder
penalizes changes to the FFHQ subject's face embedding.

## Setup

The project uses Python 3.13 and `uv`:

```bash
uv sync --group dev
```

The configured datasets live in `data/FFHQ` and `data/SNGFaces`. Download or
resume them with:

```bash
uv run python src/download.py --dataset ffhq
uv run python src/download.py --dataset sngfaces
```

Download only the AuraFace recognition model (not its bundled detection or
attribute models):

```bash
uv run python src/download.py --asset auraface
```

The artifact is pinned by repository revision and SHA-256 in
`config/assets.yaml`. AuraFace is declared Apache-2.0 by its model card; this
repository records that declaration but does not independently determine the
provenance or license of its training data.

## Train

The defaults in `config/train.yaml` use the RTX-friendly EnCo settings from the
paper implementation: batch size 1, 256px crops, `5e-5` generator/feature
learning rates, `2e-4` discriminator learning rate, and a 100+100 epoch schedule.
The first 20 epochs warm up the learning rate. Each epoch visits all 644
SNGFaces paintings once and samples FFHQ faces randomly.

```bash
uv run python -m src.train --config config/train.yaml
```

For Colab training with checkpoints stored in Google Drive, open
[notebooks/colab.ipynb](notebooks/colab.ipynb).

For an end-to-end smoke test:

```bash
uv run python -m src.train --config config/train.yaml --max-steps 1
```

Outputs are stored under `outputs/ffhq_to_sngfaces_enco/`:

- `checkpoints/latest.pt` contains G, D, the EnCo feature heads, optimizers,
  schedulers, AMP state, step, epoch, and resolved config.
- `metrics.csv` records training losses plus epoch-level KID evaluation rows.
- `early_stopping.csv` records KID and ArcFace distance every five epochs from epoch 100.
- `samples/` contains rows of source, generated, and real target images.

Early stopping saves the lowest-KID checkpoint that remains below the configured
ArcFace identity-distance ceiling as `checkpoints/best.pt`.

Set `wandb.enabled: true` to log the resolved hyperparameters, training losses,
KID, and validation ArcFace distance to Weights & Biases. Authentication uses
the standard `WANDB_API_KEY` environment variable; the Colab notebook exposes it
as an optional form field.

Resume with `--resume path/to/latest.pt`, or set `training.resume` in YAML.

## Inference

```bash
uv run python -m src.infer \
  --config config/train.yaml \
  --checkpoint outputs/ffhq_to_sngfaces_enco/checkpoints/latest.pt \
  --input data/FFHQ/00000.png \
  --output outputs/example.png
```

Launch the upload-and-translate Gradio UI with:

```bash
uv run python -m src.ui \
  --config config/train.yaml \
  --checkpoint outputs/ffhq_to_sngfaces_enco/checkpoints/best.pt
```

The input may also be a flat directory, in which case the output must be a
directory.

## Objectives

The default generator objective is:

```text
1.0 * LSGAN
+ original EnCo content/target-identity objective (2.0 / 10.0)
+ 1.0 * (1 - cosine(AuraFace(source), AuraFace(generated)))
```

AuraFace is converted from ONNX to a frozen PyTorch module so gradients reach
the generated image. The source embedding is detached, and no face detector is
used because FFHQ and its translation share aligned geometry.

## Tests

```bash
uv run pytest -m "not integration"
```

The large parity/gradient test is opt-in:

```bash
AURAFACE_MODEL=data/.cache/models/auraface/glintr100.onnx \
  uv run pytest tests/test_identity.py -m integration
```

The EnCo design is adapted with attribution from Xiuding Cai et al.'s
BSD-2-Clause implementation. AuraFace and `onnx2torch` remain under their
respective licenses.
