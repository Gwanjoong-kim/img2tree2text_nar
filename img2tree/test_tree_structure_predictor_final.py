import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from datasets import load_dataset, concatenate_datasets
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import pdb

# 기존 모듈 경로 추가
sys.path.append("/home1/kim03/myubai/IMG2TEXT_NAR/clean_narit")
from integrated_model import ModelConfig
from data_preprocessing import MyTreeStructureDataset
from train_tree_structure_predictor_aug_final import ARDecodingLightningModule  # 이미 정의된 클래스

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    gt_matrices = [torch.tensor(item["gt_matrix"]) for item in batch]
    gt_matrix_padded = nn.utils.rnn.pad_sequence(gt_matrices, batch_first=True, padding_value=0)
    return {"pixel_values": pixel_values, "gt_matrix": gt_matrix_padded}

class TestDataModule(pl.LightningDataModule):
    def __init__(self, dataset, test_cfg):
        super().__init__()
        self.dataset = dataset
        self.test_bsz = test_cfg["test_bsz"]

    def setup(self, stage=None):
        self.train_dataset = MyTreeStructureDataset(self.dataset)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.test_bsz, shuffle=False, num_workers=4, collate_fn=collate_fn)

if __name__ == "__main__":
    # Load test dataset
    ds1 = load_dataset("naver-clova-ix/cord-v1", split="test")
    ds = ds1

    # Test configuration
    test_cfg = {"test_bsz": 3}

    # Initialize data module
    data_module = TestDataModule(ds, test_cfg)
    
    # Logger for wandb
    wandb_logger = WandbLogger(
        project="tree_structure_project",
        name="test_run",
        log_model=True
    )

    # Model configuration
    config = ModelConfig(encoder_layer=[2, 2, 14, 2], input_size=[2560, 1920], window_size=12)
    # Load pre-trained model from checkpoint
    checkpoint_path = "checkpoints/with_aug_constraint/model-epoch=42-train_loss=0.14.ckpt"
    model = ARDecodingLightningModule.load_from_checkpoint(checkpoint_path, config=config)

    # Initialize trainer for testing
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,  # 테스트는 단일 GPU에서 수행
        logger=wandb_logger
    )

    # Run the test loop
    print("Starting test...")
    trainer.test(model, datamodule=data_module)
    pdb.set_trace()