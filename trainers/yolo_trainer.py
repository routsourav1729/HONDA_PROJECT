"""
YOLOv8 trainer implementation with improved multi-GPU support
"""
import os
import yaml
import torch
from pathlib import Path
from .base_trainer import BaseTrainer

class YOLOTrainer(BaseTrainer):
    """Trainer for YOLOv8 models with enhanced multi-GPU support"""
    
    def __init__(self, config_path, data_path, output_dir="runs", gpu_ids=None):
        """Initialize YOLOv8 trainer"""
        super().__init__(config_path, data_path, output_dir, gpu_ids)
        
        # Load model-specific parameters
        self.model_path = self.config.get('model', 'yolov8n.pt')
        self.epochs = self.config.get('epochs', 50)
        self.img_size = self.config.get('imgsz', 640)
        self.batch_size = self.config.get('batch_size', 16)
        self.name = self.config.get('name', 'yolov8_run')
        
        # Create output directories explicitly to avoid file not found errors
        self.run_dir = os.path.join(self.output_dir, self.name)
        os.makedirs(self.run_dir, exist_ok=True)
        
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
        """Run YOLOv8 training with improved multi-GPU support"""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Make sure output directories exist before training starts
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Extra check for multiple GPUs to create result files in advance
        if len(self.gpu_ids) > 1:
            # Create results.csv file to prevent the FileNotFoundError
            results_csv = os.path.join(self.run_dir, "results.csv")
            if not os.path.exists(results_csv):
                with open(results_csv, 'w') as f:
                    f.write("epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n")
        
        # Configure multi-GPU training
        if len(self.gpu_ids) > 1:
            device = self.gpu_ids  # Pass list of GPU IDs for multi-GPU
            print(f"Training on multiple GPUs: {self.gpu_ids}")
        else:
            device = self.gpu_ids[0] if self.gpu_ids else 'cpu'
            print(f"Training on: {device}")
        
        try:
            # Run training
            results = self.model.train(
                data=self.data_path,
                epochs=self.epochs,
                imgsz=self.img_size,
                batch=self.batch_size,
                project=self.output_dir,
                name=self.name,
                device=device,
                exist_ok=True  # Prevent errors if directory exists
            )
            return results
        except Exception as e:
            # Enhanced error handling
            if "CUDA out of memory" in str(e):
                print(f"ERROR: CUDA out of memory. Try reducing batch size or using fewer GPUs.")
            elif "DDP" in str(e) or "distributed" in str(e):
                print(f"ERROR: Distributed training issue: {e}")
                print("Try using a single GPU or updating ultralytics package.")
            else:
                print(f"ERROR: Training failed with error: {e}")
            
            # Return partial results if available
            return None
    
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