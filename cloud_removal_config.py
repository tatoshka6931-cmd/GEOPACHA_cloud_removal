#cloud_removal_config.py
import os
import random
import shutil
import logging
import numpy as np
import torch
from pathlib import Path
from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner.dataset import SemanticSegmentationSlidingWindowGeoDataset


UID = os.getuid()
GVFS_BASE = (
    f'/run/user/{UID}/gvfs/'
    f'smb-share:server=sarlserver06.cas.vanderbilt.edu,'
    f'share=sarl_commons06/Wernke_projects/GeoPACHA/'
    f'Imagery_Machine_Learning/Image_Preprocessing/Cloud_Removal_Project'
)
 
IMAGE_DIRECTORY  = f'{GVFS_BASE}/Images/'
AOI_DIRECTORY    = f'{GVFS_BASE}/AOI/'
LABELS_URI       = f'{GVFS_BASE}/Labels.geojson'
LOG_DIRECTORY    = f'{GVFS_BASE}/model/training_logs/'
OUTPUT_DIRECTORY = f'{GVFS_BASE}/model/output_v2/'
LOCAL_LABELS_DIR = '/home/kapitot/labels_output_tempfolder/'
WEIGHTS_PATH = f'{OUTPUT_DIRECTORY}cloud_model_weights.pth'    # train.py writes here; infer.py reads from here.
LOCAL_IMG_CACHE = '/home/kapitot/images_local_copy/'
 
# parameters 
BATCH_SIZE   = 8
TILE_SIZE    = 512   # 512x512 
TRAIN_STRIDE = 256   # 50% overlap during training
INFER_STRIDE = 512   # each pixel predicted once
NUM_EPOCHS   = 9
LR           = 1e-3
NUM_WORKERS  = 4

SEED         = 79
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
 
# classes
class_config = ClassConfig(
    names=['background', 'cloud'],
    colors=['black', 'white'],
    null_class='background',
)
 
 
# setup logger
def setup_logging(mode: str) -> None:
    n = len(list(Path(LOG_DIRECTORY).glob('run_*.log'))) + 1
    log_path = f'{LOG_DIRECTORY}run_{n}_{mode}.log'             
    logging.basicConfig(
        format='%(asctime)s  %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ]
    )
 
 
# normalize image to resnet50 standards to [0,1]
def normalize_img_values(image, mask=None, **kwargs):
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    return {'image': image, 'mask': mask}
 
 
# helper functions
def match_data():
    """
    Pair every AOI GeoJSON with its matching WV-2 TIF, shuffle, then split
    70 / 15 / 15 into train / val / test.
    """
    all_samples = []
    for aoi_path in Path(AOI_DIRECTORY).glob('*.geojson'):
        image_id = aoi_path.stem[9:]   # strip the 'aoi_clip_' prefix
        matches  = list(Path(IMAGE_DIRECTORY).glob(f'*{image_id}*.TIF'))
        if matches:
            all_samples.append({
                'img_id':    image_id,
                'image_uri': str(matches[0]),
                'aoi_uri':   str(aoi_path),
            })
        else:
            logging.warning(f'No TIF found for AOI {aoi_path.name} (id={image_id})')
 
    logging.info(f'Matched {len(all_samples)} image/AOI pairs')
    random.shuffle(all_samples)
 
    n = len(all_samples)
    val_data   = all_samples[:int(0.15 * n)]
    test_data  = all_samples[int(0.15 * n):int(0.45 * n)]
    train_data = all_samples[int(0.45 * n):]
    return all_samples, train_data, val_data, test_data
 
 
def build_single_ds(sample: dict, stride: int, with_labels: bool = True):
    """
    Build a SemanticSegmentationSlidingWindowGeoDataset for one image.
 
    Parameters
    ----------
    sample      : dict with 'image_uri' and 'aoi_uri'
    stride      : tile stride in pixels (pass TRAIN_STRIDE or INFER_STRIDE)
    with_labels : include label_vector_uri; set False when running on images
                  that have no ground-truth cloud annotations yet
    """
    label_kw = {}
    if with_labels:
        label_kw['label_vector_uri']              = LABELS_URI
        label_kw['label_vector_default_class_id'] = class_config.get_class_id('cloud')

    image_uri = get_local_image_copy(sample['image_uri'])   

    return SemanticSegmentationSlidingWindowGeoDataset.from_uris(
        class_config=class_config,
        image_uri=image_uri,                                
        aoi_uri=sample['aoi_uri'],
        within_aoi=True,
        image_raster_source_kw=dict(allow_streaming=True),
        size=TILE_SIZE,
        stride=stride,
        transform=normalize_img_values,
        **label_kw,
    )
 
def find_image_by_id(image_id: str) -> str:
    """Locate the full-resolution TIF for a given image ID, regardless of AOI."""
    matches = list(Path(IMAGE_DIRECTORY).glob(f'*{image_id}*.TIF'))
    if not matches:
        raise FileNotFoundError(f'No image found for id={image_id} in {IMAGE_DIRECTORY}')
    if len(matches) > 1:
        logging.warning(f'Multiple images matched id={image_id}: {matches}; using first')
    return str(matches[0])


def build_full_image_ds(image_uri: str, stride: int):
    """
    Sliding-window dataset over the WHOLE image — no AOI restriction.
    Omitting aoi_uri/within_aoi means windows cover the full raster extent.
    """
    image_uri=get_local_image_copy(image_uri)

    return SemanticSegmentationSlidingWindowGeoDataset.from_uris(
        class_config=class_config,
        image_uri=image_uri,
        image_raster_source_kw=dict(allow_streaming=True),
        size=TILE_SIZE,
        stride=stride,
        transform=normalize_img_values,
    )


def get_local_image_copy(image_uri: str) -> str:
    """Copy a raster to local disk once, so sliding-window reads don't hit Samba per-tile."""
    Path(LOCAL_IMG_CACHE).mkdir(parents=True, exist_ok=True)
    local_path = Path(LOCAL_IMG_CACHE) / Path(image_uri).name

    if local_path.exists() and local_path.stat().st_size != Path(image_uri).stat().st_size:
        logging.warning(f'Cached copy of {local_path.name} looks truncated — re-copying.')
        local_path.unlink()

    if not local_path.exists():
        logging.info(f'Caching {image_uri} locally…')
        shutil.copyfile(image_uri, local_path)

    return str(local_path)
