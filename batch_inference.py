from pathlib import Path
from cloud_removal_inference import make_learner, run_inference_on_image_path
import os

UID = os.getuid()
GVFS_BASE = (
    f'/run/user/{UID}/gvfs/'
    f'smb-share:server=sarlserver06.cas.vanderbilt.edu,'
    f'share=sarl_commons06/Wernke_projects/GeoPACHA/'
    f'Imagery_Machine_Learning/Image_Preprocessing/Cloud_Removal_Project'
)
INPUT_DIRECTORY = f'{GVFS_BASE}/images_to_process'  

learner = make_learner()
for image_path in sorted(Path(INPUT_DIRECTORY).glob('*.TIF')):
    run_inference_on_image_path(learner, str(image_path), img_id=image_path.stem)