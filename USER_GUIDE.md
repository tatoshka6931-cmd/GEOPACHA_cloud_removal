# GEOPACHA Cloud Removal — User Guide

This guide explains what the pipeline does, how to run it, and how to adapt it
for your own imagery. For the underlying methodology, band-normalization
rationale, and training-data decisions, see `GeoPACHA_documentation.docx` in
this repo — read it before changing normalization or training data.

## 1. What this pipeline does

Given 8-band WorldView-2 satellite imagery, the model predicts which pixels
are cloud-covered, so those areas can be masked out or excluded from
downstream analysis. It's a semantic segmentation model (2 classes:
`background`, `cloud`) built on an FPN/ResNet-50 backbone (via
[`AdeelH/pytorch-fpn`](https://github.com/AdeelH/pytorch-fpn)), trained with
[RasterVision](https://docs.rastervision.io/).

Output for each processed image: a raster cloud mask plus a vectorized
GeoJSON of cloud polygons.

## 2. Repository structure

| File | Purpose |
|---|---|
| `cloud_removal_config.py` | Paths, hyperparameters, band normalization, dataset-building helpers. **Edit this first.** |
| `cloud_removal_training.py` | Trains the model, saves weights + a model bundle + validation prediction plots. |
| `cloud_removal_inference.py` | Runs inference on one image (full-image or AOI-restricted) and produces mask + vector output. |
| `batch_inference.py` | Loops full-image inference over every `.TIF` in a directory. |
| `run_detached_process_guide.md` | How to run long training/inference jobs in `tmux` so they survive a crashed IDE/SSH session. |
| `GeoPACHA_documentation.docx` | Full methodology write-up — read before touching normalization, training data, or hard negatives. |

## 3. Environment setup

The repo doesn't ship a `requirements.txt`/`environment.yml`, so you'll need
to build the environment yourself. At minimum:

- Python with a `rastervision` conda/pip environment (RasterVision core +
  `rastervision_pytorch_learner`)
- PyTorch (with CUDA if you have a GPU)
- `numpy`
- Internet access on first run — `cloud_removal_training.py` and
  `cloud_removal_inference.py` both pull the FPN model architecture from
  GitHub via `torch.hub.load('AdeelH/pytorch-fpn:0.3', ...)`

SARL-specific: the lab's existing environment is a conda env literally named
`rastervision` (see `run_detached_process_guide.md` for activation).

## 4. Configuring `cloud_removal_config.py`

### Paths (SARL-specific — you will need to change these)

```python
GVFS_BASE = (
    f'/run/user/{UID}/gvfs/'
    f'smb-share:server=sarlserver06.cas.vanderbilt.edu,'
    f'share=sarl_commons06/Wernke_projects/GeoPACHA/'
    f'Imagery_Machine_Learning/Image_Preprocessing/Cloud_Removal_Project'
)
```

This points at a Samba share mounted via GVFS on the SARL lab workstations —
it will not exist outside that network. Replace `GVFS_BASE` (and the derived
`IMAGE_DIRECTORY`, `AOI_DIRECTORY`, `LABELS_URI`, `LOG_DIRECTORY`,
`OUTPUT_DIRECTORY`) with paths that make sense for your own storage —
local disk, a different network share, cloud storage, etc. `LOCAL_LABELS_DIR`
and `LOCAL_IMG_CACHE` are local scratch directories used to avoid streaming
directly off a network mount during training/inference; keep them local to
whatever machine is doing the compute.

### Data expectations

- `IMAGE_DIRECTORY`: full-resolution `.TIF` images.
- `AOI_DIRECTORY`: GeoJSON files named `aoi_clip_<image_id>.geojson` — one
  per image, used to restrict training/eval tiling to a labeled area.
- `LABELS_URI`: a single GeoJSON of cloud polygon labels covering all
  images (labels are matched to images by spatial overlap at dataset-build
  time, not by file).

`match_data()` pairs each AOI file to an image by matching the ID portion of
the filename (`aoi_clip_<image_id>.geojson` → `*<image_id>*.TIF`), then
splits pairs 15% val / 30% test / 55% train — at the image level, so tiles
from the same image never leak across splits.

### Hyperparameters worth knowing about

| Constant | What it controls |
|---|---|
| `TILE_SIZE` | Chip size fed to the model (512×512). |
| `TRAIN_STRIDE` / `INFER_STRIDE` | Sliding-window stride — smaller stride during training (overlap, more samples), larger during inference (each pixel predicted ~once). |
| `NUM_EPOCHS`, `LR`, `BATCH_SIZE` | Standard training params. |
| `NUM_WORKERS` | Set to `1` in config, but forced to `0` in `cloud_removal_inference.py` — parallel workers were unstable reading over the Samba mount. Raise this if you're not reading off a network share. |
| `DENOISE_FACTOR` | Passed to RasterVision's `PolygonVectorOutputConfig` — higher values remove more small cloud polygons from the vector output. |
| `BAND_MIN` / `BAND_MAX` | Per-band clip/normalization ranges, sampled from 7 training images. **These are specific to this WorldView-2 dataset.** If you bring your own imagery (different sensor, different radiance range), recompute these from your own data — don't reuse them as-is. |

### Class config

```python
class_config = ClassConfig(names=['background', 'cloud'], null_class='background')
```

Only two classes exist — there's no separate "no-data" class for
image-edge/nodata pixels, so they fall into `background`. If you vectorize
background output, clip it against the raster's valid-data footprint first
or you'll get spurious polygons around image edges.

## 5. Training

```bash
python cloud_removal_training.py
```

This builds train/val/test dataloaders from `match_data()`, trains for
`NUM_EPOCHS`, then:
1. Saves weights to `WEIGHTS_PATH` (`{OUTPUT_DIRECTORY}/cloud_model_weights.pth`)
2. Attempts to save a full model bundle (non-fatal if it fails)
3. Attempts to save validation prediction plots (non-fatal if it fails)

Logs are written to `LOG_DIRECTORY` and echoed to stdout, numbered
`run_<n>_train.log`.

## 6. Inference

Two modes, both in `cloud_removal_inference.py`:

**Full-image (no AOI restriction) — the default:**
Set `TARGET_IMAGE_ID` near the top of the file to the image ID you want to
run, then:
```bash
python cloud_removal_inference.py
```

**AOI-restricted batch (e.g. to evaluate on the held-out test set):**
Set `TARGET_IMAGE_ID = None`, which falls back to running inference over
`test_data` from `match_data()` (or switch to `all_samples` in the code if
you want every matched image).

**Batch over a whole directory of images (no AOI, no labels needed):**
```bash
python batch_inference.py
```
This loops every `.TIF` in `{GVFS_BASE}/images_to_process` and runs full-image
inference on each — update `INPUT_DIRECTORY` at the top of the file for your
own setup.

Each run produces, per image: a discrete raster label mask and a
`<img_id>_clouds.geojson` polygon file, saved locally first
(`LOCAL_LABELS_DIR`) then copied to `{OUTPUT_DIRECTORY}/predictions/<img_id>/`.

## 7. Running long jobs unattended

Training/inference on large imagery can run for hours and has been known to
crash an IDE or SSH session. Use `tmux` (see `run_detached_process_guide.md`
for the full walkthrough):

```bash
tmux new -s train
conda activate rastervision
python cloud_removal_training.py
# detach: Ctrl+B, then D
tmux attach -t train   # reattach later
nvidia-smi -l 1         # monitor GPU usage in another pane/session
```

## 8. Adapting this to your own project — checklist

1. Set up an environment with RasterVision + PyTorch (see §3).
2. Point `GVFS_BASE`/`IMAGE_DIRECTORY`/`AOI_DIRECTORY`/`LABELS_URI`/
   `LOG_DIRECTORY`/`OUTPUT_DIRECTORY` at your own storage.
3. Get your imagery and AOI/label GeoJSONs into the expected layout (§4).
4. Recompute `BAND_MIN`/`BAND_MAX` from a sample of your own images —
   don't reuse the values in this repo unless you're using the same sensor
   and processing level.
5. If your imagery isn't 8-band, or you have more than 2 classes, update
   `in_channels` and `num_classes` in `load_model()` in **both**
   `cloud_removal_training.py` and `cloud_removal_inference.py`, and update
   `class_config` in `cloud_removal_config.py` to match.
6. Confirm you have internet access on the machine that will run training/
   inference (needed for the `torch.hub.load` model download), or
   pre-download/cache the FPN architecture if working offline.
7. Start with a short `NUM_EPOCHS` run to confirm the pipeline runs
   end-to-end before committing to a full training run.
