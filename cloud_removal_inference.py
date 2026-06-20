#cloud_removal_inference.py
import shutil
import logging
from pathlib import Path
import torch
from rastervision.core.data import SemanticSegmentationLabels
from rastervision.pytorch_learner import (
    SemanticSegmentationGeoDataConfig,
    SolverConfig,
    SemanticSegmentationLearnerConfig,
    SemanticSegmentationLearner,
)

# edit cloud_removal_config.py file
from cloud_removal_config import (
    class_config,
    setup_logging,
    match_data,
    build_single_ds,
    find_image_by_id,
    build_full_image_ds,
    BATCH_SIZE,
    TILE_SIZE,
    INFER_STRIDE,
    LR,
    NUM_WORKERS,
    OUTPUT_DIRECTORY,
    LOCAL_LABELS_DIR,
    WEIGHTS_PATH,
)
# lower for inference, else it crashes
NUM_WORKERS=0

TARGET_IMAGE_ID = "1030010058BF5300" # edit if want to run on target image
 
# minimal learner setup for inference 
class InferenceLearner(SemanticSegmentationLearner):
    def setup_data(self, distributed: bool = False):
        pass
 
 
# load existing model
def load_model():
    logging.info('Loading model architecture (FPN/ResNet50, 8-band)…')
    model = torch.hub.load(
        'AdeelH/pytorch-fpn:0.3',
        'make_fpn_resnet',
        name='resnet50',
        fpn_type='panoptic',
        num_classes=2,
        fpn_channels=128,
        in_channels=8,
        out_size=(TILE_SIZE, TILE_SIZE),
        pretrained=True)
    state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logging.info('loading model complete')
    return model
 
 
def make_learner():
    model = load_model()
    data_cfg    = SemanticSegmentationGeoDataConfig(
        class_config=class_config,
        num_workers=NUM_WORKERS,)

    solver_cfg  = SolverConfig(batch_sz=BATCH_SIZE, num_epochs=1, lr=LR)
    learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)
 
    return InferenceLearner(
        cfg=learner_cfg,
        output_dir=OUTPUT_DIRECTORY,
        model=model,
        training=False)
 
 
def run_inference_on_image(learner: InferenceLearner, sample: dict):
    img_id = sample['img_id']
    logging.info(f'[{img_id}]: running inference')
    ds = build_single_ds(sample, stride=INFER_STRIDE, with_labels=False)
    logging.info(f'[{img_id}]  {len(ds)} tiles to predict')
 
    with torch.no_grad():
        predictions = learner.predict_dataset(
            ds,
            raw_out=True,
            numpy_out=True,
            progress_bar=True,
            dataloader_kw=dict(num_workers=NUM_WORKERS, pin_memory=False),
        )

    pred_labels = SemanticSegmentationLabels.from_predictions(
        ds.windows,
        predictions,
        smooth=True,
        extent=ds.scene.extent,
        num_classes=len(class_config),
    )
 
    save_and_copy_predictions(pred_labels,ds,img_id)
 

def save_and_copy_predictions(pred_labels, ds, img_id):
    local_out = Path(LOCAL_LABELS_DIR)/img_id
    local_out.mkdir(parents=True, exist_ok=True)
    pred_labels.save(
        uri=str(local_out),
        crs_transformer=ds.scene.raster_source.crs_transformer,
        class_config=class_config,
        discrete_output=True,
    )
    logging.info(f'[{img_id}] Labels saved locally @ {local_out}')

    server_out = Path(OUTPUT_DIRECTORY)/'predictions'/img_id
    server_out.mkdir(parents=True, exist_ok=True)
    for f in local_out.iterdir():
        shutil.copyfile(f, server_out / f.name)
    logging.info(f'[{img_id}] copied to server @ {server_out}')


def run_inference_on_full_image(learner: InferenceLearner, image_id: str):
    image_uri = find_image_by_id(image_id)
    logging.info(f'[{image_id}]: running inference on FULL image (no AOI)')
    ds = build_full_image_ds(image_uri, stride=INFER_STRIDE)
    logging.info(f'[{image_id}]  {len(ds)} tiles to predict')

    with torch.no_grad():
        predictions = learner.predict_dataset(
            ds, raw_out=True, numpy_out=True, progress_bar=True,
            dataloader_kw=dict(num_workers=NUM_WORKERS, pin_memory=False),
        )

    pred_labels = SemanticSegmentationLabels.from_predictions(
        ds.windows, predictions, smooth=True,
        extent=ds.scene.extent, num_classes=len(class_config),
    )
    save_and_copy_predictions(pred_labels, ds, image_id)




def run_inference():
    setup_logging('infer')
    learner = make_learner()

    if TARGET_IMAGE_ID:
        # single full-image run, no AOI restriction
        try:
            run_inference_on_full_image(learner, TARGET_IMAGE_ID)
        except Exception as exc:
            logging.error(f'[{TARGET_IMAGE_ID}] Failed: {exc}', exc_info=True)
    else:
        # existing AOI-restricted batch behavior
        all_samples, _, _, test_data = match_data()
        targets = test_data     # select only to test model
        #targets = all_samples  # select to apply model to all samples

        logging.info(f'Running inference on {len(targets)} images:')
        for sample in targets:
            try:
                run_inference_on_image(learner, sample)
            except Exception as exc:
                logging.error(f'[{sample["img_id"]}] Failed: {exc}', exc_info=True)

    logging.info('Inference complete.')
 



 
if __name__ == '__main__':
    run_inference()
