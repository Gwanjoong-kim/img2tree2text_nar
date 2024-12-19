import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datasets import load_dataset, concatenate_datasets
from pytorch_lightning.loggers import WandbLogger

# Import your custom modules here
from tree_decoder.tree_structure_decoder_effi import ARDecodingModel
from donut_model import SwinEncoder
from data_preprocessing import MyTreeStructureDataset
from integrated_model import ModelConfig

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    gt_matrices = [torch.tensor(item["gt_matrix"]) for item in batch]
    gt_matrix_padded = nn.utils.rnn.pad_sequence(gt_matrices, batch_first=True, padding_value=-1)
    return {"pixel_values": pixel_values, "gt_matrix": gt_matrix_padded}

class ARDecodingLightningModule(pl.LightningModule):
    def __init__(self, config, hidden_dim=1024, num_heads=8, num_layers=1, max_nodes=800, max_token_length=400, lr=1e-4):
        super(ARDecodingLightningModule, self).__init__()
        self.encoder = SwinEncoder(**config.to_dict())
        self.model = ARDecodingModel(hidden_dim, num_heads, num_layers, max_nodes, max_token_length)
        self.lr = lr
        self.max_nodes = max_nodes
        self.max_token_length = max_token_length

        # Embeddings with padding_idx
        self.node_embedding = nn.Embedding(num_embeddings=4, embedding_dim=256, padding_idx=-1)
        self.parent_embedding = nn.Embedding(num_embeddings=max_nodes, embedding_dim=512, padding_idx=-1)
        self.token_length_embedding = nn.Embedding(num_embeddings=max_token_length, embedding_dim=256, padding_idx=-1)

        # Combined layer
        self.combined_layer = nn.Linear(256 + 512 + 256, hidden_dim)

        # Loss functions with ignore_index for padding
        self.type_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.parent_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.token_length_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pixel_values, decoder_input, node_types_so_far):
        self.encoder_output = self.encoder(pixel_values).view(pixel_values.size(0), -1, 1024)
        return self.model(self.encoder_output, decoder_input, node_types_so_far)

    def training_step(self, batch, batch_idx):
        batch_size, seq_len, _ = batch["gt_matrix"].size()
        device = batch["gt_matrix"].device

        pixel_values = batch["pixel_values"]
        gt_matrix = batch["gt_matrix"]

        # Target tensors
        target_node_type = gt_matrix[:, :, 0].long()
        target_parent_index = gt_matrix[:, :, 1].long()
        target_token_length = gt_matrix[:, :, 2].long()

        # Initial inputs
        initial_node_type = torch.zeros(batch_size, dtype=torch.long, device=device)
        initial_parent_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        initial_token_length = torch.zeros(batch_size, dtype=torch.long, device=device)

        node_type_embedding = self.node_embedding(initial_node_type)
        parent_idx_embedding = self.parent_embedding(initial_parent_idx)
        token_length_embedding = self.token_length_embedding(initial_token_length)

        # Initial decoder input
        combined_initial_input = torch.cat([node_type_embedding, parent_idx_embedding, token_length_embedding], dim=-1)
        decoder_input = self.combined_layer(combined_initial_input).unsqueeze(1)

        node_types_so_far = initial_node_type.unsqueeze(1)

        total_loss = 0.0
        total_type_loss = 0.0
        total_parent_loss = 0.0
        total_token_length_loss = 0.0

        end_token = -1

        for t in range(seq_len):
            outputs = self.forward(pixel_values, decoder_input, node_types_so_far)
            node_type_logits, parent_node_logits, token_length_logits, node_type_pred, parent_index_pred, token_length_pred, _ = outputs

            # Check for end token
            if (node_type_pred == end_token).any():
                print(f"Stopping early at step {t} because one of the outputs hit the end token (-1).")
                break

            # Update node_types_so_far
            node_types_so_far = torch.cat([node_types_so_far, node_type_pred.unsqueeze(1)], dim=1)

            # Calculate losses
            type_loss = self.type_loss_fn(node_type_logits.squeeze(1), target_node_type[:, t])
            parent_loss = self.parent_loss_fn(parent_node_logits.squeeze(1), target_parent_index[:, t])
            token_length_loss = self.token_length_loss_fn(token_length_logits.squeeze(1), target_token_length[:, t])

            total_loss += type_loss + parent_loss + token_length_loss
            total_type_loss += type_loss.item()
            total_parent_loss += parent_loss.item()
            total_token_length_loss += token_length_loss.item()

            # Prepare next decoder input
            next_node_embedding = self.node_embedding(node_type_pred)
            next_parent_embedding = self.parent_embedding(parent_index_pred)
            next_token_length_embedding = self.token_length_embedding(token_length_pred)
            next_input = torch.cat([next_node_embedding, next_parent_embedding, next_token_length_embedding], dim=-1)
            decoder_input = self.combined_layer(next_input).unsqueeze(1)

        # Average losses
        avg_loss = total_loss / seq_len
        avg_type_loss = total_type_loss / seq_len
        avg_parent_loss = total_parent_loss / seq_len
        avg_token_length_loss = total_token_length_loss / seq_len

        # Logging
        if self.trainer.is_global_zero:
            self.log('train_loss', avg_loss)
            self.log('type_loss', avg_type_loss)
            self.log('parent_loss', avg_parent_loss)
            self.log('token_length_loss', avg_token_length_loss)

        return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_cfg):
        super().__init__()
        self.dataset = dataset
        self.train_bsz = train_cfg["train_bsz"]
        self.eval_bsz = train_cfg["eval_bsz"]

    def setup(self, stage=None):
        self.train_dataset = MyTreeStructureDataset(self.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_bsz, shuffle=True, num_workers=4, collate_fn=collate_fn)

if __name__ == "__main__":
    # Load datasets
    cord_1 = load_dataset("naver-clova-ix/cord-v1", split="train")
    cord_2 = load_dataset("naver-clova-ix/cord-v2", split="train")
    ds = concatenate_datasets([cord_1, cord_2])

    # Configuration
    config = ModelConfig(encoder_layer=[2, 2, 14, 2], input_size=[2560, 1920], window_size=12)
    model = ARDecodingLightningModule(config)
    train_cfg = {
        "train_bsz": 1,
        "eval_bsz": 1
    }
    data_module = MyDataModule(ds, train_cfg)
    wandb_logger = WandbLogger(project="tree_structure_project")

    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=10,
        accumulate_grad_batches=4,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision='16-mixed',
        accelerator='gpu',
        devices=4,
        logger=wandb_logger
    )

    trainer.fit(model, data_module)