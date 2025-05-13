"""
Base trainer class that defines common training functionality
"""
import os
import yaml
import torch
from pathlib import Path
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Abstract base class for all model trainers"""
    
    def __init__(self, config_path, data_path, output_dir="runs", gpu_ids=None):
        """
        Initialize base trainer
        
        Args:
            config_path: Path to model configuration YAML
            data_path: Path to dataset configuration YAML
            output_dir: Directory to save training outputs
            gpu_ids: List of GPU IDs to use for training, or None for CPU
        """
        self.config_path = config_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.gpu_ids = gpu_ids if gpu_ids is not None else []
        
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        with open(data_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Set up device
        self.setup_device()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_device(self):
        """Set up training device (CPU/GPU)"""
        if not self.gpu_ids:
            self.device = torch.device('cpu')
            print("Using CPU for training")
            return
            
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        
        # Validate GPU IDs
        for gpu_id in self.gpu_ids:
            if gpu_id >= num_gpus:
                raise ValueError(f"Requested GPU {gpu_id} but only {num_gpus} GPUs available")
        
        # Set CUDA_VISIBLE_DEVICES for multiple GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        
        # For single-GPU case
        if len(self.gpu_ids) == 1:
            self.device = torch.device(f'cuda:{0}')
            print(f"Using GPU: {self.gpu_ids[0]}")
        # For multi-GPU case
        else:
            self.device = torch.device('cuda')
            print(f"Using GPUs: {self.gpu_ids}")
    
    @abstractmethod
    def setup_model(self):
        """Set up model architecture - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def setup_dataloader(self):
        """Set up data loaders - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def train(self):
        """Run training process - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate(self):
        """Run validation - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def save_model(self, path):
        """Save model to disk - must be implemented by subclasses"""
        pass
    
    def load_model(self, path):
        """Load model from disk - may be overridden by subclasses"""
        raise NotImplementedError("Model loading not implemented")