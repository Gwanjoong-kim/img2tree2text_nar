import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pdb
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datasets import load_dataset, concatenate_datasets
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import torch.nn.functional as F
import torch.optim as optim

# Import your custom modules here
sys.path.append("/home1/kim03/myubai/IMG2TEXT_NAR/clean_narit")
from donut_model import SwinEncoder
from integrated_model import ModelConfig

from data_preprocessing import MyTreeStructureDataset
from tree_structure_decoder_effi import ARDecodingModel

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    gt_matrices = [torch.tensor(item["gt_matrix"]) for item in batch]
    gt_matrix_padded = nn.utils.rnn.pad_sequence(gt_matrices, batch_first=True, padding_value=0)
    return {"pixel_values": pixel_values, "gt_matrix": gt_matrix_padded}

class SaveAfterEpochCallback(Callback):
    def __init__(self, save_every_n_epochs=10, save_path="model_checkpoint"):
        self.save_every_n_epochs = save_every_n_epochs  # N 에포크마다 저장
        self.save_path = save_path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # epoch이 10의 배수이고, step이 0일 때 저장
        if trainer.current_epoch % self.save_every_n_epochs == 0 and batch_idx == 0:
            save_path = f"{self.save_path}/epoch_{trainer.current_epoch}_step_{batch_idx}.ckpt"
            trainer.save_checkpoint(save_path)
            print(f"Model saved at {save_path}")

class ARDecodingLightningModule(pl.LightningModule):
    def __init__(self, model, hidden_dim=16, lr=3e-5):
        super().__init__()    
        config = ModelConfig(encoder_layer=[2, 2, 14, 2], input_size=[2560, 1920], window_size=12)
        self.encoder = SwinEncoder(**config.to_dict())
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.lr = lr
        EOS_NODE_TYPE, EOS_PARENT_IDX, EOS_TOKEN_LENGTH = 5, 101, 91
        self.type_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=EOS_NODE_TYPE, ignore_index=0)
        self.parent_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=EOS_PARENT_IDX, ignore_index=0)
        self.token_length_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=EOS_TOKEN_LENGTH, ignore_index=0)

    def forward(self, pixel_values, input_node_type_seq, input_parent_index_seq, input_token_length_seq):
        encoder_output = self.encoder(pixel_values).view(pixel_values.size(0), -1, self.model.hidden_dim)
        # print(encoder_output.size(),flush=True)
        output = self.model(encoder_output, input_node_type_seq, input_parent_index_seq, input_token_length_seq)
        return output

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        gt_matrix = batch["gt_matrix"]

        target_node_type = gt_matrix[:, :, 0].long()
        target_parent_index = gt_matrix[:, :, 1].long()
        target_token_length = gt_matrix[:, :, 2].long()

        target_node_type = target_node_type.masked_fill(target_node_type <= 0, 0)
        target_parent_index = target_parent_index.masked_fill(target_parent_index <= 0, 0)
        target_token_length = target_token_length.masked_fill(target_token_length <= 0, 0)
        
        batch_size = pixel_values.size(0)
        start_node_type = torch.full((batch_size, 1), 1, dtype=torch.long, device=self.device)
        start_parent_index = torch.full((batch_size, 1), 1, dtype=torch.long, device=self.device)
        start_token_length = torch.full((batch_size, 1), 1, dtype=torch.long, device=self.device)

        input_node_type_seq = torch.cat([start_node_type, target_node_type[:, :-1]], dim=1)
        input_parent_index_seq = torch.cat([start_parent_index, target_parent_index[:, :-1]], dim=1)
        input_token_length_seq = torch.cat([start_token_length, target_token_length[:, :-1]], dim=1)
        
        # print(f"pixel_values: {pixel_values.size()}, input_node_type_seq: {input_node_type_seq.size()}, input_parent_index_seq: {input_parent_index_seq.size()}, input_token_length_seq: {input_token_length_seq.size()}",flush=True)
        node_type_logits, parent_node_logits, token_length_logits = self.forward(
            pixel_values, input_node_type_seq, input_parent_index_seq, input_token_length_seq
        )

        # Compute loss
        type_loss = self.loss_fn(node_type_logits.view(-1, node_type_logits.size(-1)), target_node_type.view(-1))
        parent_loss = self.loss_fn(parent_node_logits.view(-1, parent_node_logits.size(-1)), target_parent_index.view(-1))
        token_length_loss = self.loss_fn(token_length_logits.view(-1, token_length_logits.size(-1)), target_token_length.view(-1))
        total_loss = type_loss + parent_loss + token_length_loss
        # Logging
        self.log('train_loss', total_loss)
        self.log('type_loss', type_loss)
        self.log('parent_loss', parent_loss)
        self.log('token_length_loss', token_length_loss)
        self.log('type_accuracy', self.type_accuracy, on_step=False, on_epoch=True)
        self.log('parent_accuracy', self.parent_accuracy, on_step=False, on_epoch=True)
        self.log('token_length_accuracy', self.token_length_accuracy, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

# DataModule 정의
class TreeDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            MyTreeStructureDataset(self.dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

# 메인 실행 코드
if __name__ == "__main__":
    # 데이터셋 로드
    ds1 = load_dataset("naver-clova-ix/cord-v1", split="train")
    ds2 = load_dataset("naver-clova-ix/cord-v2", split="train")
    ds = concatenate_datasets([ds1, ds2])

    # ARDecodingModel 초기화
    decoding_model = ARDecodingModel()
    model = ARDecodingLightningModule(decoding_model)
    # 데이터 모듈 초기화
    data_module = TreeDataModule(dataset=ds, batch_size=1)
    
    # wandb_logger = WandbLogger(
    #     project="tree_structure_project",
    #     name="training_run_with_accuracy",
    #     log_model=True
    # )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/with_eos/',
        filename='model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min'
    )

    # 트레이너 설정 및 학습
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=4,  # Change this based on the number of GPUs you have
        # logger = wandb_logger,
        precision="16-mixed",  
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="gloo"),
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0  # Optional gradient clipping
    )
    trainer.fit(model, data_module)