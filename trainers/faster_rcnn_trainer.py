"""
Faster R-CNN trainer implementation 
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from .base_trainer import BaseTrainer

class FasterRCNNTrainer(BaseTrainer):
    """Trainer for Faster R-CNN models"""
    
    def __init__(self, config_path, data_path, output_dir="runs", gpu_ids=None, pretrained=True):
        """Initialize Faster R-CNN trainer"""
        super().__init__(config_path, data_path, output_dir, gpu_ids)
        
        # Load model-specific parameters
        self.backbone = self.config.get('backbone', 'resnet50')
        self.lr = self.config.get('learning_rate', 0.005)
        self.momentum = self.config.get('momentum', 0.9)
        self.weight_decay = self.config.get('weight_decay', 0.0005)
        self.epochs = self.config.get('epochs', 20)
        self.batch_size = self.config.get('batch_size', 4)
        self.img_size = self.config.get('img_size', 800)
        self.pretrained = pretrained
        self.name = self.config.get('name', 'faster_rcnn_run')
        
        # Initialize model, optimizer, and dataloaders
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Create model
        self.setup_model()
    
    def setup_model(self):
        """Set up Faster R-CNN model"""
        try:
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            
            # Get number of classes from dataset config
            num_classes = self.data_config.get('nc', 0) + 1  # +1 for background class
            
            # Initialize model with pretrained weights if requested
            self.model = fasterrcnn_resnet50_fpn(pretrained=self.pretrained)
            
            # Replace the classifier with a new one for custom number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # Move model to device
            self.model.to(self.device)
            
            # Use DataParallel if multiple GPUs
            if len(self.gpu_ids) > 1:
                self.model = nn.DataParallel(self.model)
                
            # Set up optimizer
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.SGD(
                params, 
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
            
            # Set up learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=3,
                gamma=0.1
            )
            
            print(f"Initialized Faster R-CNN model with {num_classes} classes")
            
        except ImportError:
            raise ImportError("Failed to import required modules. Please install torchvision.")
    
    def setup_dataloader(self):
        """Set up data loaders for training and validation"""
        try:
            from dataloader.idd import IDDDataset
            import torchvision.transforms as T
            
            # Define transforms
            def get_transform(train):
                transforms = []
                transforms.append(T.ToTensor())
                if train:
                    transforms.append(T.RandomHorizontalFlip(0.5))
                return T.Compose(transforms)
            
            # Create datasets
            train_dataset = IDDDataset(
                root_dir=self.data_config.get('path'),
                split='train',
                img_size=self.img_size,
                transform=get_transform(train=True)
            )
            
            val_dataset = IDDDataset(
                root_dir=self.data_config.get('path'),
                split='val',
                img_size=self.img_size,
                transform=get_transform(train=False)
            )
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=self._collate_fn
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self._collate_fn
            )
            
            print(f"Created data loaders with {len(train_dataset)} training and {len(val_dataset)} validation samples")
            
        except ImportError:
            raise ImportError("Failed to import required modules. Please check your dataloader implementation.")
    
    @staticmethod
    def _collate_fn(batch):
        """Custom collate function for object detection batches"""
        return tuple(zip(*batch))
    
    def train_one_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for images, targets in self.train_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)} 
                      for t in targets]
            
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
        
        return total_loss / len(self.train_loader)
    
    def train(self):
        """Run Faster R-CNN training"""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
            
        if self.train_loader is None or self.val_loader is None:
            self.setup_dataloader()
        
        print(f"Starting training for {self.epochs} epochs")
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_one_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(self.output_dir, f"{self.name}_best.pth"))
        
        # Save final model
        self.save_model(os.path.join(self.output_dir, f"{self.name}_final.pth"))
        print("Training complete")
        
        return {"train_loss": train_loss, "val_loss": val_loss}
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)} 
                          for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, path=None):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not initialized or trained")
            
        if path is None:
            path = os.path.join(self.output_dir, f"{self.name}_final.pth")
        
        # Save model state
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
            
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        if self.model is None:
            self.setup_model()
            
        state_dict = torch.load(path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
            
        print(f"Loaded model from {path}")
        return self.model