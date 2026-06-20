#cloud_removal_training.py
import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset

from rastervision.pytorch_learner import (
    SemanticSegmentationGeoDataConfig,
    SolverConfig,
    SemanticSegmentationLearnerConfig,
    SemanticSegmentationLearner,
)

from cloud_removal_config import (
    class_config,
    setup_logging,
    match_data,
    build_single_ds,
    BATCH_SIZE,
    TILE_SIZE,
    TRAIN_STRIDE,
    NUM_EPOCHS,
    LR,
    NUM_WORKERS,
    OUTPUT_DIRECTORY,
    WEIGHTS_PATH,
)


# custom learner to add pytorch dataloaders instead of rastervision ones

class OverrideLearner(SemanticSegmentationLearner):
    def __init__(self,
        cfg: SemanticSegmentationLearnerConfig,
        output_dir: str,
        tdl: DataLoader,
        vdl: DataLoader,
        testdl: DataLoader,
        model: torch.nn.Module = None):

        self.tdl    = tdl
        self.vdl    = vdl
        self.testdl = testdl
        super().__init__(cfg, output_dir=output_dir, model=model, training=True)

    def setup_data(self, distributed: bool = False):
        self.setup_ddp_params()
        self.train_dl = self.tdl
        self.valid_dl = self.vdl
        self.test_dl  = self.testdl
        self.train_ds = self.tdl.dataset   
        self.test_ds  = self.testdl.dataset
        self.batch_sz = BATCH_SIZE

        w = self.cfg.solver.class_loss_weights
        if w != None:
            self.loss_weights = torch.tensor(w, dtype=torch.float)
        else:
            self.loss_weights=None


# model
def load_model():
    model = torch.hub.load(
        'AdeelH/pytorch-fpn:0.3',
        'make_fpn_resnet',
        name='resnet50',
        fpn_type='panoptic',
        num_classes=2,
        fpn_channels=128,
        in_channels=8,
        out_size=(TILE_SIZE, TILE_SIZE),
        pretrained=True,
    )
    logging.info('Model loaded (FPN/ResNet50, 8-band input, 2 output classes)')
    return model

def make_dataloader(samples: list, shuffle: bool):
    ds = ConcatDataset([build_single_ds(s, stride=TRAIN_STRIDE) for s in samples])
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )


# learner 

def build_learner(train_dl: DataLoader, val_dl: DataLoader, test_dl: DataLoader):
    data_cfg = SemanticSegmentationGeoDataConfig(
        class_config=class_config,
        num_workers=NUM_WORKERS)
    
    solver_cfg = SolverConfig(
        batch_sz=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        class_loss_weights=[1.0, 10.0])
    
    learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)

    return OverrideLearner(
        cfg=learner_cfg,
        output_dir=OUTPUT_DIRECTORY,
        tdl=train_dl,
        vdl=val_dl,
        testdl=test_dl,
        model=load_model(),
    )


def run_training():
    setup_logging('train')

    _, train_data, val_data, test_data = match_data()

    logging.info('Building dataloaders…')
    train_dl = make_dataloader(train_data, shuffle=True)
    val_dl   = make_dataloader(val_data,   shuffle=False)   
    test_dl  = make_dataloader(test_data,  shuffle=False)   
    logging.info(
        f'Tile counts — train: {len(train_dl.dataset)} val: {len(val_dl.dataset)}  test: {len(test_dl.dataset)}')

    learner = build_learner(train_dl, val_dl, test_dl)

    logging.info(f'Starting training for {NUM_EPOCHS} epochs')
    learner.train(epochs=NUM_EPOCHS)
    logging.info('Training loop complete')

    torch.save(learner.model.state_dict(), WEIGHTS_PATH)
    logging.info(f'Weights saved @ {WEIGHTS_PATH}')

    try:
        learner.save_model_bundle()
        logging.info(f'Model bundle saved @ {OUTPUT_DIRECTORY}')
    except Exception as exc:
        logging.warning(f'save_model_bundle failed (non-fatal): {exc}')

    try:
        learner.plot_predictions(split='valid', show=False)
        logging.info('Validation prediction plots saved')
    except Exception as exc:
        logging.warning(f'plot_predictions failed (non-fatal): {exc}')


if __name__ == '__main__':
    run_training()