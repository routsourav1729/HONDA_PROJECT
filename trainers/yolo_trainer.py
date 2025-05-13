"""
YOLOv8 trainer implementation
"""
import os
import yaml
import torch
from pathlib import Path
from .base_trainer import BaseTrainer

class YOLOTrainer(BaseTrainer):
    """Trainer for YOLOv8 models"""
    
    def __init__(self, config_path, data_path, output_dir="runs", gpu_ids=None):
        """Initialize YOLOv8 trainer"""
        super().__init__(config_path, data_path, output_dir, gpu_ids)
        
        # Load model-specific parameters
        self.model_path = self.config.get('model', 'yolov8n.pt')
        self.epochs = self.config.get('epochs', 50)
        self.img_size = self.config.get('imgsz', 640)
        self.batch_size = self.config.get('batch_size', 16)
        self.name = self.config.get('name', 'yolov8_run')
        
        # Initialize model
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Set up YOLOv8 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"Loaded YOLOv8 model: {self.model_path}")
        except ImportError:
            raise ImportError("Failed to import 'ultralytics'. Please install it with: pip install ultralytics")
    
    def setup_dataloader(self):
        """
        YOLOv8 trainer doesn't need to explicitly create dataloaders
        as they are handled internally by the YOLO.train() method
        """
        pass
    
    def train(self):
        """Run YOLOv8 training"""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # For multiple GPUs, device is already set in CUDA_VISIBLE_DEVICES
        if len(self.gpu_ids) > 1:
            device = list(range(len(self.gpu_ids)))
        else:
            device = 0 if self.gpu_ids else 'cpu'
        
        # Run training
        results = self.model.train(
            data=self.data_path,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            project=self.output_dir,
            name=self.name,
            device=device
        )
        
        return results
    
    def validate(self, weights=None):
        """Run validation on trained model"""
        if weights is None and self.model is None:
            raise ValueError("Either provide weights path or train model first")
        
        # Load weights if provided
        if weights is not None:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            
        # Run validation
        results = self.model.val(
            data=self.data_path,
            imgsz=self.img_size,
            batch=self.batch_size
        )
        
        return results
    
    def save_model(self, path=None):
        """
        Save model to disk
        Note: YOLOv8 automatically saves checkpoints during training
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained")
            
        if path is None:
            path = os.path.join(self.output_dir, f"{self.name}_final.pt")
            
        # Export the model
        self.model.export(format="pt", save=True)
        print(f"Model exported to {path}")
    
    def load_model(self, path):
        """Load YOLOv8 model from disk"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            print(f"Loaded YOLOv8 model from {path}")
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")