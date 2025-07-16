#!/usr/bin/env python3
"""
Custom Grounding DINO Training Script
Fine-tune Grounding DINO for warehouse pallet detection
"""

import os
import sys
import yaml
import torch
import logging
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection,
    TrainingArguments,
    Trainer
)
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WarehousePalletDataset(torch.utils.data.Dataset):
    """Custom dataset for warehouse pallet detection"""
    
    def __init__(self, data_dir: str, processor, config: Dict):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.config = config
        
        # Load dataset
        dataset_file = self.data_dir / "dataset.json"
        with open(dataset_file, 'r') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        logger.info(f"Loaded {len(self.images)} images from {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        
        # Load image
        image_path = self.data_dir / "images" / item['image_path']
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare text prompts
        texts = item['texts'] if item['texts'] else ["pallet on warehouse floor"]
        text = ". ".join(texts)
        
        # Process with processor
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        )
        
        # Prepare targets (bounding boxes)
        boxes = torch.tensor(item['boxes'], dtype=torch.float32)  # [N, 4] normalized
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'boxes': boxes,
            'labels': torch.ones(len(boxes), dtype=torch.long)  # All are "pallet" class
        }

class GroundingDINOTrainer:
    """Custom trainer for Grounding DINO fine-tuning"""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and processor
        self.setup_model()
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup training components
        self.setup_training()
        
        # Setup logging
        self.setup_logging()
    
    def setup_model(self):
        """Initialize model and processor"""
        model_name = self.config['model']['pretrained_model']
        
        logger.info(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['hardware']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
        
        logger.info("✅ Model loaded successfully")
    
    def setup_datasets(self):
        """Setup training and validation datasets"""
        data_root = Path(self.config['dataset']['data_root'])
        
        # Training dataset
        train_dir = data_root / "train"
        self.train_dataset = WarehousePalletDataset(
            train_dir, self.processor, self.config
        )
        
        # Validation dataset
        val_dir = data_root / "val"
        self.val_dataset = WarehousePalletDataset(
            val_dir, self.processor, self.config
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['dataloader_num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['validation']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['dataloader_num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            collate_fn=self.collate_fn
        )
        
        logger.info(f"✅ Datasets loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Handle variable number of boxes per image
        all_boxes = [item['boxes'] for item in batch]
        all_labels = [item['labels'] for item in batch]
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'boxes': all_boxes,
            'labels': all_labels
        }
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.config['training']['lr_milestones'],
            gamma=self.config['training']['lr_gamma']
        )
        
        # Mixed precision scaler
        if self.config['training']['use_amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info("✅ Training components initialized")
    
    def setup_logging(self):
        """Setup logging and experiment tracking"""
        # Create output directories
        self.output_dir = Path(self.config['output']['model_dir'])
        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                entity=self.config['logging']['wandb_entity'],
                config=self.config,
                name=f"grounding_dino_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        logger.info("✅ Logging setup complete")
    
    def compute_loss(self, outputs, targets):
        """Compute custom loss for Grounding DINO"""
        # This is a simplified loss computation
        # In practice, you'd use the model's built-in loss computation
        
        # Extract predictions
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        
        # Compute classification loss
        classification_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), 
                                                   targets['labels'].view(-1))
        
        # Compute box regression loss (simplified)
        box_loss = nn.MSELoss()(pred_boxes, targets['boxes'])
        
        # Combine losses
        total_loss = (self.config['training']['loss_weights']['classification'] * classification_loss + 
                     self.config['training']['loss_weights']['bbox_regression'] * box_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'box_loss': box_loss
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.config['training']['use_amp']:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    losses = self.compute_loss(outputs, batch)
                    loss = losses['total_loss']
            else:
                outputs = self.model(**batch)
                losses = self.compute_loss(outputs, batch)
                loss = losses['total_loss']
            
            # Backward pass
            if self.config['training']['use_amp']:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config['logging']['log_every_n_steps'] == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Loss: {loss.item():.4f}")
                
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': self.current_epoch,
                        'global_step': self.global_step
                    })
        
        return sum(epoch_losses) / len(epoch_losses)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                losses = self.compute_loss(outputs, batch)
                val_losses.append(losses['total_loss'].item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        if self.config['logging']['use_wandb']:
            wandb.log({
                'val_loss': avg_val_loss,
                'epoch': self.current_epoch
            })
        
        return avg_val_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Best model saved: {best_path}")
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            if epoch % self.config['validation']['eval_every_n_epochs'] == 0:
                val_loss = self.validate()
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f} {'(Best!)' if is_best else ''}")
                
                # Save checkpoint
                if epoch % self.config['training']['save_every_n_epochs'] == 0:
                    self.save_checkpoint(is_best)
            
            # Update learning rate
            self.scheduler.step()
        
        # Save final model
        final_model_path = Path(self.config['output']['final_model_dir']) / "final_model.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), final_model_path)
        
        logger.info("✅ Training complete!")
        
        if self.config['logging']['use_wandb']:
            wandb.finish()

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train custom Grounding DINO model')
    parser.add_argument('--config', type=str, default='cv/training/configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = GroundingDINOTrainer(args.config)
    
    if args.resume:
        # Load checkpoint
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"✅ Resumed from checkpoint: {args.resume}")
    
    trainer.train()

if __name__ == "__main__":
    main()
