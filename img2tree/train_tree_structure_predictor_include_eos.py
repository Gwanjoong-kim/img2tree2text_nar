import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from datasets import load_dataset, concatenate_datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

# Import your custom modules here
sys.path.append("/home1/kim03/myubai/IMG2TEXT_NAR/clean_narit")
from donut_model import SwinEncoder
from integrated_model import ModelConfig
from data_preprocessing import MyTreeStructureDataset
from tree_structure_decoder_effi import ARDecodingModel
from pytorch_lightning.loggers import WandbLogger
import pdb

# Set environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    gt_matrices = [torch.tensor(item["gt_matrix"]) for item in batch]
    gt_matrix_padded = nn.utils.rnn.pad_sequence(gt_matrices, batch_first=True, padding_value=0)
    return {"pixel_values": pixel_values, "gt_matrix": gt_matrix_padded}

class ARDecodingLightningModule(pl.LightningModule):
    def __init__(self, config, hidden_dim=1024, num_heads=8, num_layers=4, max_nodes=100, max_token_length=90, lr=3e-5):
        super(ARDecodingLightningModule, self).__init__()
        self.save_hyperparameters()

        self.encoder = SwinEncoder(**config.to_dict())
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.model = ARDecodingModel(hidden_dim, num_heads, num_layers, max_nodes, max_token_length)
        self.lr = lr
        self.max_nodes = max_nodes
        self.max_token_length = max_token_length

        # Embeddings with padding_idx=0
        self.node_embedding = nn.Embedding(num_embeddings=5 + 1, embedding_dim=256, padding_idx=0)  # 0번 인덱스를 패딩으로 사용
        self.parent_embedding = nn.Embedding(num_embeddings=max_nodes + 1, embedding_dim=512, padding_idx=0)
        self.token_length_embedding = nn.Embedding(num_embeddings=max_token_length + 1, embedding_dim=256, padding_idx=0)

        # Combined layer
        self.combined_layer = nn.Linear(256 + 512 + 256, hidden_dim)

        # Loss functions with ignore_index=0 for padding
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, pixel_values, node_type_seq, parent_index_seq, token_length_seq):
        # encoder_output을 한 번만 계산
        encoder_output = self.encoder(pixel_values).view(pixel_values.size(0), -1, 1024)
        return self.model(encoder_output, node_type_seq, parent_index_seq, token_length_seq, node_types_so_far, apply_constraints=True)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        gt_matrix = batch["gt_matrix"]

        # Target tensors
        target_node_type = gt_matrix[:, :, 0].long()
        target_parent_index = gt_matrix[:, :, 1].long()
        target_token_length = gt_matrix[:, :, 2].long()

        # Handle padding
        target_node_type = target_node_type.masked_fill(target_node_type <= 0, 0)
        target_parent_index = target_parent_index.masked_fill(target_parent_index <= 0, 0)
        target_token_length = target_token_length.masked_fill(target_token_length <= 0, 0)

        # Define START tokens (assuming index 1 is a valid start token)
        START_NODE_TYPE = 1
        START_PARENT_INDEX = 1
        START_TOKEN_LENGTH = 1

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        start_node_type = torch.full((batch_size, 1), START_NODE_TYPE, dtype=torch.long, device=device)
        start_parent_index = torch.full((batch_size, 1), START_PARENT_INDEX, dtype=torch.long, device=device)
        start_token_length = torch.full((batch_size, 1), START_TOKEN_LENGTH, dtype=torch.long, device=device)

        # Prepare input sequences for teacher forcing
        input_node_type_seq = torch.cat([start_node_type, target_node_type[:, :-1]], dim=1)
        input_parent_index_seq = torch.cat([start_parent_index, target_parent_index[:, :-1]], dim=1)
        input_token_length_seq = torch.cat([start_token_length, target_token_length[:, :-1]], dim=1)

        # Forward pass
        encoder_output = self.encoder(pixel_values).view(batch_size, -1, 1024)
        node_types_so_far = start_node_type  # Initialize with start token
        
        # 모델 호출
        outputs = self.model(
            encoder_output,
            node_type_seq=input_node_type_seq,
            parent_index_seq=input_parent_index_seq,
            token_length_seq=input_token_length_seq,
            node_types_so_far=node_types_so_far,
            apply_constraints=False
        )
        
        node_type_logits = outputs[0]
        parent_node_logits = outputs[1]
        token_length_logits = outputs[2]      
        
        # node_type 정확도 계산 및 저장
        node_type_preds = torch.argmax(node_type_logits, dim=-1).view(-1)  # 예측값
        target_node_type_flat = target_node_type.view(-1)  # 실제값
        self.node_type_accuracy = (node_type_preds == target_node_type_flat).float().mean().item()  # 정확도 계산

        # parent_node 정확도 계산 및 저장
        parent_node_preds = torch.argmax(parent_node_logits, dim=-1).view(-1)  # 예측값
        target_parent_index_flat = target_parent_index.view(-1)  # 실제값
        self.parent_accuracy = (parent_node_preds == target_parent_index_flat).float().mean().item()  # 정확도 계산

        # token_length 정확도 계산 및 저장
        token_length_preds = torch.argmax(token_length_logits, dim=-1).view(-1)  # 예측값
        target_token_length_flat = target_token_length.view(-1)  # 실제값
        self.token_length_accuracy = (token_length_preds == target_token_length_flat).float().mean().item()  # 정확도 계산

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
        self.log('node_type_accuracy', self.node_type_accuracy)
        self.log('parent_accuracy', self.parent_accuracy)
        self.log('token_length_accuracy', self.token_length_accuracy)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)  # Reduce LR every 2 epochs
        return [optimizer], [scheduler]

class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, train_bsz=14, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.train_bsz = train_bsz
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_bsz, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

if __name__ == "__main__":
    # Load datasets
    ds1 = load_dataset("naver-clova-ix/cord-v1", split="train")
    ds2 = load_dataset("naver-clova-ix/cord-v2", split="train")
    ds = concatenate_datasets([ds1, ds2])

    # Prepare dataset
    train_dataset = MyTreeStructureDataset(ds)

    # Configuration
    config = ModelConfig(encoder_layer=[2, 2, 14, 2], input_size=[2560, 1920], window_size=12)
    model = ARDecodingLightningModule(config)

    # Prepare DataModule
    data_module = MyDataModule(train_dataset, train_bsz=14)
    
    wandb_logger = WandbLogger(
        project="tree_structure_project",
        name="training_run_with_accuracy",
        log_model=True
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/with_eos/',
        filename='model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=4,  # Change this based on the number of GPUs you have
        logger = wandb_logger,
        precision=32,  
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="gloo"),
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0  # Optional gradient clipping
    )

    # Training
    trainer.fit(model, data_module)