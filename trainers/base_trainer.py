"""
Base trainer class that defines common training functionality with improved multi-GPU support
"""
import os
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Abstract base class for all model trainers with enhanced multi-GPU support"""
    
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        with open(data_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Set up device
        self.setup_device()
    
    def setup_device(self):
        """Set up training device (CPU/GPU) with better multi-GPU support"""
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        
        if not self.gpu_ids:
            self.device = torch.device('cpu')
            print("Using CPU for training")
            return
            
        # Validate GPU IDs
        valid_gpu_ids = []
        for gpu_id in self.gpu_ids:
            if gpu_id >= num_gpus:
                print(f"Warning: Requested GPU {gpu_id} but only {num_gpus} GPUs available. Skipping.")
            else:
                valid_gpu_ids.append(gpu_id)
        
        if not valid_gpu_ids:
            print(f"Warning: No valid GPUs found among requested IDs {self.gpu_ids}. Falling back to CPU.")
            self.device = torch.device('cpu')
            self.gpu_ids = []
            return
        
        # Update gpu_ids to only include valid ones
        self.gpu_ids = valid_gpu_ids
        
        # Set CUDA_VISIBLE_DEVICES for multiple GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        
        # For single-GPU case
        if len(self.gpu_ids) == 1:
            self.device = torch.device(f'cuda:{0}')  # Always use cuda:0 since we set CUDA_VISIBLE_DEVICES
            print(f"Using GPU: {self.gpu_ids[0]}")
        # For multi-GPU case
        else:
            self.device = torch.device('cuda')
            print(f"Using {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
    
    def initialize_distributed(self, local_rank=0):
        """Initialize distributed training if needed"""
        if len(self.gpu_ids) <= 1:
            return False  # Not using distributed training
        
        # Initialize distributed process group
        try:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            return True
        except Exception as e:
            print(f"Failed to initialize distributed training: {e}")
            print("Falling back to single GPU training")
            self.gpu_ids = [self.gpu_ids[0]]
            self.device = torch.device(f'cuda:{0}')
            return False
    
    def create_run_directories(self, name):
        """Create all necessary directories for training run"""
        run_dir = os.path.join(self.output_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Create subdirectories that might be needed
        for subdir in ['weights', 'logs', 'checkpoints']:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
        
        return run_dir
    
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
    
    def is_main_process(self):
        """Check if this is the main process in distributed training"""
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    
    def cleanup_distributed(self):
        """Clean up distributed training resources if needed"""
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()