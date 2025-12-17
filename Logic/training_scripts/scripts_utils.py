import torch
from google.cloud import storage
import random
import numpy as np
import os
import time
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F


def upload_to_gcp(bucket_name, source_file_name, destination_blob_name, max_retries=3):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    for attempt in range(max_retries + 1):
        try:
            blob.upload_from_filename(source_file_name)
            print(f"File {source_file_name} uploaded to {destination_blob_name}.")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Upload failed.")
                raise


def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2  # In MB
    cached = torch.cuda.memory_reserved() / 1024**2  # In MB
    print(f"GPU Memory Cached: {cached:.2f} MB, GPU Memory Allocated: {allocated:.2f} MB")


def handle_tuple(param):
    """Converts tuple parameters to a string-friendly format."""
    if isinstance(param, (tuple, list)):
        return "_".join(map(str, param))  # Converts (1, 100) to "1_100"
    return str(param)  # Returns non-tuple params as they are


def generate_checkpoint_name(save_path, epoch, model_name=None, **kwargs):
    # Format hyperparameters into the filename
    kwargs_names = "_".join([f"{key[:2]}_{handle_tuple(value)}" for key, value in kwargs.items()]).replace(".", "_")
    if model_name:
        filename = f"ep_{epoch}_{model_name}_{kwargs_names}.pt"
    else:
        filename = f"ep_{epoch}_{kwargs_names}.pt"

    # Check if file exists and append a number if it does
    if os.path.exists(os.path.join(save_path, filename)):
        base_name, ext = os.path.splitext(filename)
        counter = 1
        # Append a number to the file name if it exists
        while os.path.exists(os.path.join(save_path, f"{base_name}_{counter}{ext}")):
            counter += 1
        filename = f"{base_name}_{counter}{ext}"

    return os.path.join(save_path, filename)


def set_seed(seed=42):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU


def checkpointed_forward(module, *inputs):
    return checkpoint(module, *inputs)


class FlexibleMarginCosineEmbeddingLoss(nn.Module):
    def __init__(self):
        super(FlexibleMarginCosineEmbeddingLoss, self).__init__()

    def forward(self, x1, x2, y, margin):
        """
        Computes the flexible margin cosine embedding loss.

        Parameters:
        - x1: Tensor of shape (batch_size, embedding_dim), first input
        - x2: Tensor of shape (batch_size, embedding_dim), second input
        - y: Tensor of shape (batch_size,), labels (1 for similar, -1 for dissimilar)
        - margin: Tensor of shape (batch_size,), dynamic margin values per example

        Returns:
        - Loss value (scalar)
        """
        cos_sim = F.cosine_similarity(x1, x2, dim=-1)

        loss = torch.where(
            y == 1,
            1 - cos_sim,  # Standard cosine embedding loss when y == 1
            torch.clamp(cos_sim - margin, min=0),  # Flexible margin for y == -1
        )

        return loss.mean()  # Return average loss across the batch
