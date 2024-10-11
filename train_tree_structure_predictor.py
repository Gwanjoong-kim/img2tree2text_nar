import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datasets import load_dataset, concatenate_datasets
import wandb
from pytorch_lightning.loggers import WandbLogger
from tree_structure_decoder import ARDecodingModel
from donut_model import SwinEncoder
from data_preprocessing import MyTreeStructureDataset
from integrated_model import ModelConfig

def collate_fn(batch):
    # batch: List of dictionaries with keys 'pixel_values' and 'gt_matrix'
    
    # 'pixel_values' 리스트 생성 (패딩 처리 없이 그대로 사용)
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    gt_matrices = []
    for item in batch:
        gt_matrix_tensor = torch.tensor(item["gt_matrix"])
        gt_matrices.append(gt_matrix_tensor)
    
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
        
        # Embeddings for each prediction variable
        self.node_embedding = nn.Embedding(num_embeddings=4, embedding_dim=256)  # Assuming 4 types of nodes
        self.parent_embedding = nn.Embedding(num_embeddings=max_nodes, embedding_dim=512)  # Maximum possible parent nodes
        self.token_length_embedding = nn.Embedding(num_embeddings=max_token_length, embedding_dim=256)  # Maximum token length
        
        # Linear layer to combine the embeddings to the decoder input dimension
        self.combined_layer = nn.Linear(256 + 512 + 256, hidden_dim)  # Combines the concatenated embeddings into hidden_dim
        
        # 손실 함수
        self.type_loss_fn = nn.CrossEntropyLoss()
        self.parent_loss_fn = nn.CrossEntropyLoss()
        self.token_length_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, pixel_values, decoder_input, node_types_so_far):
        self.encoder_output = self.encoder(pixel_values).view(pixel_values.size(0), 4800, 1024)
        return self.model(self.encoder_output, decoder_input, node_types_so_far)
    
    def training_step(self, batch, batch_idx):
        # batch: (batch_size, seq_len, num_features)
        # 필요한 입력과 타깃 추출
        batch_size, seq_len, _ = batch["gt_matrix"].size()
        
        device = batch["gt_matrix"].device
        
        pixel_values = batch["pixel_values"]
        gt_matrix = batch["gt_matrix"]
        
        # 타깃 텐서 추출
        target_node_type = gt_matrix[:, :, 0].long().to(device)          # (batch_size, seq_len)
        target_parent_index = gt_matrix[:, :, 1].long().to(device)       # (batch_size, seq_len)
        target_token_length = gt_matrix[:, :, 2].long().to(device)       # (batch_size, seq_len)
        
        # 패딩 토큰에 대한 마스크 생성 (여기서는 0이 패딩 토큰이라고 가정)
        # 실제 패딩 토큰 인덱스에 따라 수정 필요
        padding_mask = (target_node_type == -1)  # 패딩 토큰 인덱스가 -1인 경우
                        
        initial_node_type = torch.tensor([0], device=device)  
        initial_parent_idx = torch.tensor([0], device=device) 
        initial_token_length = torch.tensor([0], device=device)
        
        node_type_embedding = self.node_embedding(initial_node_type)
        parent_idx_embedding = self.parent_embedding(initial_parent_idx)
        token_length_embedding = self.token_length_embedding(initial_token_length)

        # 처음 입력은 초기 임베딩값들로 결합
        combined_initial_input = torch.cat([node_type_embedding, parent_idx_embedding, token_length_embedding], dim=-1)
        decoder_input = self.combined_layer(combined_initial_input)
        decoder_input = decoder_input.repeat(batch_size,1)
        decoder_input = decoder_input.unsqueeze(1)
        
        node_types_so_far = torch.zeros(batch_size, 1, dtype=torch.long, device=device) 
        
        all_node_types = []
        all_parent_indices = []
        all_token_lengths = []
        
        total_loss = 0.0
        
        end_token = -1
        
        # 최대 시퀀스 길이만큼 반복
        for t in range(seq_len):
            outputs = self.forward(pixel_values, decoder_input, node_types_so_far)
            all_node_types.append(outputs[3])
            all_parent_indices.append(outputs[4])
            all_token_lengths.append(outputs[5])
            
            # print(f"Node Type : {outputs[3]}", flush=True)
            # print(f"Parent Index : {outputs[4]}", flush=True)
            # print(f"Token Length : {outputs[5]}", flush=True)
            # print(f"Seq len : {seq_len}", flush=True)
             
            if (outputs[3] == end_token).any() or (outputs[4] == end_token).any() or (outputs[5] == end_token).any():
                print(f"Stopping early at step {t} because one of the outputs hit the end token (-1).")
                break      
            # outputs: [node_type_logits, parent_node_logits, token_length_logits, node_type, parent_node_index, token_length, node_types_so_far]

            # node_types_so_far 업데이트: 이전 스텝의 node_type을 이어 붙임
            node_types_so_far = torch.cat([node_types_so_far, outputs[3]], dim=1)  # 기존에 이어 붙임
                        
            type_loss = self.type_loss_fn(outputs[0][:, 0, :].float(), target_node_type[:, t]).mean()
            parent_loss = self.parent_loss_fn(outputs[1][:, 0, :].float(), target_parent_index[:, t]).mean()
            token_length_loss = self.token_length_loss_fn(outputs[2][:, 0, :].float(), target_token_length[:, t].long())

            total_loss = type_loss + parent_loss + token_length_loss
            
            # Concatenating the embeddings and projecting them to hidden_dim
            next_input = self.combined_layer(torch.cat([self.node_embedding(outputs[3]), self.parent_embedding(outputs[4]), self.token_length_embedding(outputs[5])], dim=-1))
            decoder_input = next_input
            
        all_node_types = torch.cat(all_node_types, dim=1)
        all_parent_indices = torch.cat(all_parent_indices, dim=1)
        all_token_lengths = torch.cat(all_token_lengths, dim=1)    
        
        # 메모리 사용량을 로그로 기록
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Memory Allocated: {memory_allocated:.2f} MB", flush=True)
        print(f"Memory Reserved: {memory_reserved:.2f} MB", flush=True)
        
        if (batch_idx + 1) % 100 == 0:
        
            result_tensor = torch.cat([all_node_types, all_parent_indices, all_token_lengths], dim=0)
            saved_tensor = result_tensor.cpu()
            torch.save(saved_tensor, "result_tensor.pt")
            
        # 평균 손실 계산
        loss = total_loss / seq_len
        if self.global_rank == 0:
            self.log('train_loss', loss)
            wandb.log({"type_loss": type_loss, "parent_loss": parent_loss, "token_length_loss": token_length_loss})
            wandb.log({"train_loss": loss})
        
        torch.cuda.empty_cache()
        return loss
    
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
        # Split the dataset into train and validation sets if necessary
        self.train_dataset = MyTreeStructureDataset(self.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_bsz, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
if __name__ == "__main__":
    
    # Configuration
    cord_1= load_dataset("naver-clova-ix/cord-v1", split="train")
    cord_2 = load_dataset("naver-clova-ix/cord-v2", split="train")
    
    ds = concatenate_datasets([cord_1, cord_2])
    
    config = ModelConfig(encoder_layer=[2,2,14,2], input_size=[2560, 1920], window_size=12)
    model = ARDecodingLightningModule(config)
    train_cfg = {
        "train_bsz": 1,
        "eval_bsz": 1
    }
    data_module = MyDataModule(ds, train_cfg)
    wandb_logger = WandbLogger(project="tree_structure_project")

    trainer = pl.Trainer(max_epochs=10, 
                         accumulate_grad_batches=4,
                         strategy=DDPStrategy(find_unused_parameters=True), 
                         precision=16,
                         devices=4,
                         logger=wandb_logger)
    
    trainer.fit(model, data_module)