import os  # 추가
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from datasets import load_dataset
from transformers import XLMRobertaTokenizer
import torch
from torch.utils.data import DataLoader
from util import JSONParseEvaluator
from data_preprocessing import MyDataset
from integrated_model import ModelConfig, IntegratedModel

class LitModel(pl.LightningModule):
    def __init__(self, model, criterion, evaluator, tokenizer, config):
        super(LitModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.config = config
        self.save_hyperparameters()
    
    def forward(self, img):
        # Forward pass
        return self.model(img)
    
    def training_step(self, batch, batch_idx):
        img = batch['img'].squeeze(1)
        gt = batch['gt']  # gt: (batch_size, target_seq_len)
        
        # Forward pass
        output = self.forward(img)  # output["real_res"]
        
        # 로그 소프트맥스 적용 (CTCLoss는 로그 확률을 기대합니다)
        log_probs = torch.nn.functional.log_softmax(output["real_res"], dim=-1)  # (batch_size, T, C)
        
        T = log_probs.size(0)
        batch_size = log_probs.size(1)
        C = log_probs.size(2)        
    
        # Input lengths
        input_lengths = torch.full(size=(batch_size,), fill_value=T, dtype=torch.long).to(self.device)

        # Target lengths
        target_lengths = torch.sum(gt != self.tokenizer.pad_token_id, dim=1).to(self.device)

        # Flatten targets and remove padding tokens
        targets = gt[gt != self.tokenizer.pad_token_id].view(-1).to(self.device)
        
        # print(f"log_probs_shape: {log_probs.shape}")
        # print(f"targets_shape: {targets.shape}")
        # print(f"input_lengths_shape: {input_lengths.shape}")
        # print(f"target_lengths_shape: {target_lengths.shape}")
        # raise
        
        # CTC Loss 계산
        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
        print(f"Batch {batch_idx}: Loss calculated - {loss.item()}")
        
        # 예측 결과 디코딩 (greedy decoding)
        with torch.no_grad():
            # 가장 높은 확률의 인덱스를 선택
            _, predicted_indices = torch.max(log_probs, dim=-1)  # (T, batch_size)
            predicted_indices = predicted_indices.permute(1, 0)  # (batch_size, T)
            
            blank_token_id = 1
            
            pred_texts_before_decoding = self.tokenizer.batch_decode(predicted_indices.cpu().numpy(), skip_special_tokens=False)
            print(f"Batch {batch_idx}: Predicted text before CTC decoding: {pred_texts_before_decoding}")            
            
            # Decode sequences
            pred_sequences = []
            for indices in predicted_indices:
                indices = indices.cpu().numpy()

                # Print raw predicted indices for debugging
                print(f"Raw predicted indices: {indices}")

                # Remove consecutive duplicates
                indices = np.concatenate(([indices[0]], indices[1:][indices[1:] != indices[:-1]]))

                # Remove blank tokens
                indices = indices[indices != blank_token_id]

                pred_sequences.append(indices.tolist())

            # Decode target sequences (remove padding)
            gt_sequences = []
            for tgt in gt:
                tgt = tgt[tgt != self.tokenizer.pad_token_id]
                gt_sequences.append(tgt.cpu().numpy().tolist())

            # Convert token IDs to text
            pred_texts = self.tokenizer.batch_decode(pred_sequences, skip_special_tokens=True)
            target_texts = self.tokenizer.batch_decode(gt_sequences, skip_special_tokens=True)

            # Print predicted texts before and after decoding
            print(f"Batch {batch_idx}: Predicted text before CTC decoding: {self.tokenizer.batch_decode(predicted_indices.cpu().numpy(), skip_special_tokens=False)}")
            print(f"Batch {batch_idx}: Predicted sequences after CTC decoding: {pred_sequences}")
            # 정확도 계산
            score = self.evaluator.cal_acc(pred_texts, target_texts)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_score", score, prog_bar=True, logger=True)
        self.log(f"learnable_query", float(output["learnable_query"].float().mean().item()), prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config["lr"], eps=self.config["adam_eps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
        return [optimizer], [scheduler]

class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset, tokenizer, model_cfg, train_cfg):
        super().__init__()
        self.dataset = dataset
        self.train_bsz = train_cfg["train_bsz"]
        self.eval_bsz = train_cfg["eval_bsz"]
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        # Split the dataset into train and validation sets if necessary
        self.train_dataset = MyDataset(self.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_bsz, shuffle=True)

def main():
    
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "8142"
    # os.environ["NCCL_NET_GDR_LEVEL"] = "4"
    # os.environ["OMP_NUM_THREADS"] = str(int(os.cpu_count() // torch.cuda.device_count()))
    
    # Load tokenizer and dataset
    my_tokenizer = XLMRobertaTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
    sk = load_dataset("naver-clova-ix/synthdog-ko", split="train")
    # Define model configuration
    config = ModelConfig(encoder_layer=[2,2,14,2], input_size=[2560, 1920], window_size=12)

    # Instantiate the model and other components
    model = IntegratedModel(config, "vocab.txt")
    criterion = torch.nn.CTCLoss(blank=my_tokenizer.pad_token_id, zero_infinity=True)    
    evaluator = JSONParseEvaluator()

    # Training configuration
    training_config = {
        "train_bsz": 2,
        "eval_bsz": 1,
        "epoch": 10,
        "seed": 2024,
        "lr": 1e-4,
        "adam_eps": 1e-8,
    }
    # Lightning DataModule
    data_module = MyDataModule(sk, my_tokenizer, train_cfg=training_config, model_cfg=config)

    # Initialize Lightning model
    lit_model = LitModel(model, criterion, evaluator, my_tokenizer, training_config)
    
    wandb_logger = WandbLogger(log_model='all')
    
    tb_logger = TensorBoardLogger("logs/", name="my_model")   

    trainer = pl.Trainer(
        max_epochs=training_config["epoch"],
        accelerator="gpu",
        # strategy="ddp_spawn",  # 또는 "ddp"
        # devices=torch.cuda.device_count(),
        devices=[1],
        num_nodes=1,
        precision=16,
        accumulate_grad_batches=8,
        log_every_n_steps=1,
        logger=wandb_logger
    )

    # 모델 학습 시작
    trainer.fit(lit_model, datamodule=data_module)
    
    wandb_logger.finish()  # Finish the run

if __name__ == "__main__":
    main()