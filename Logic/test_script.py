import argparse
import os
import json
import random
import time
import neptune
import numpy as np
import torch
import torch.optim as optim
import logging

from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR

from Logic.data_loaders import create_dataloaders, evaluate_model_by_attribute, ContrastiveLoss
from Logic.NN_makers import make_siamese_vgg, make_siamese_dists, make_siamese_resnet
from Logic.nn_models_utils import count_parameters

NEPTUNE_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYTQ2MmQ1YS1mNTc0LTRkMDgtYWU1My02MTQ0MWIyNDdlNzUifQ=="


path = "/Users/orenm/BlenderShaderProject/data/"
datasets_path = os.path.join(path, "datasets/")

log_filename = os.path.join(path, "training_log.txt")
if os.path.exists(log_filename):
    # If file exists, open it in append mode
    log_file_handler = logging.FileHandler(log_filename, mode="a")
else:
    # If the file doesn't exist, create a new one
    log_file_handler = logging.FileHandler(log_filename, mode="w")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), log_file_handler],
)


def train(
    batch_size: int = 32,
    num_epochs: int = 4,
):
    auc_log_perc = 0.3
    loss_log_perc = 0.01
    logging.info(f"Starting a new run. {batch_size}, {num_epochs}")

    file_path = os.path.join(datasets_path, "texture_cls_pairs.json")
    with open(file_path, "rb") as json_file:
        data = json.load(json_file)

    logging.info(f'dataset size: {len(data)}')

    model = make_siamese_resnet((6,128), final_emb=128)

    logging.info(f"Model params: {count_parameters(model)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device")
    model = model.to(device)

    run = neptune.init_run(
        project="oren.matar/BlenderShaders",
        api_token=NEPTUNE_KEY,
    )

    criterion = torch.nn.CosineEmbeddingLoss(margin=0.3)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        '/tmp/model.pt',
    )

    run["test"].append(4)
    run.stop()


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Train the model with different hyperparameters.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs to train (default: 4).")

    # Parse the arguments
    args = parser.parse_args()

    # Call the train function with the parsed arguments
    train(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
