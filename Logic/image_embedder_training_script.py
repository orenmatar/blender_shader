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
images_path = os.path.join(path, "images/")
models_path = os.path.join(path, "models/")
db_path = os.path.join(path, "DB/")

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


def set_seed(seed=42):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU


def handle_tuple(param):
    """Converts tuple parameters to a string-friendly format."""
    if isinstance(param, (tuple, list)):
        return "_".join(map(str, param))  # Converts (1, 100) to "1_100"
    return str(param)  # Returns non-tuple params as they are


def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2  # In MB
    cached = torch.cuda.memory_reserved() / 1024**2  # In MB
    logging.info(f"GPU Memory Cached: {cached:.2f} MB, GPU Memory Allocated: {allocated:.2f} MB")


def generate_checkpoint_name(save_path, epoch, **kwargs):
    # Format hyperparameters into the filename
    kwargs_names = "_".join([f"{key[:2]}_{handle_tuple(value)}" for key, value in kwargs.items()]).replace(".", "_")
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


def train(
    sample_perc: float,
    layers_to_take: tuple[int],
    learning_rate: float,
    model_name: str,
    final_emb: int,
    pool_type: str,
    scheduler_name: str,
    loss_type: str,
    resize: bool,
    test_perc=0.1,
    batch_size: int = 32,
    num_epochs: int = 4,
):
    auc_log_perc = 0.3
    loss_log_perc = 0.01
    logging.info(f"Starting a new run")

    file_path = os.path.join(path, "texture_cls_pairs.json")
    with open(file_path, "rb") as json_file:
        data = json.load(json_file)

    def add_labels(pairs, *labels):
        return [(x[0], x[1], *labels) for x in pairs]

    dataset = []
    for pair_type in [
        "different_pairs_random",
        "different_pairs_cluster",
        "cat_numeric_pairs",
        "important_params_pairs",
    ]:
        dataset.extend(add_labels(data[pair_type], 0, pair_type))

    dataset.extend(add_labels(data["similar_pairs"], 1, "similar_pairs"))
    sample_size = int(sample_perc * len(dataset))
    sampled_dataset = random.sample(dataset, k=sample_size)
    logging.info(f"Sampled dataset with {len(sampled_dataset)} points")

    avg_pool = pool_type == "avg"
    if model_name == "vgg":
        model = make_siamese_vgg(layers_to_take, final_emb=final_emb, use_avg_pool=avg_pool)
    elif model_name == "resnet":
        model = make_siamese_resnet(layers_to_take, final_emb=final_emb, use_avg_pool=avg_pool)
    elif model_name == "dists":
        model = make_siamese_dists(final_emb=final_emb)
    else:
        raise NotImplemented

    logging.info(f"Model params: {count_parameters(model)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device")
    model = model.to(device)

    train_loader, test_loader = create_dataloaders(
        sampled_dataset, images_path, test_size=test_perc, batch_size=batch_size, num_workers=4, resize=resize
    )
    auc_log_interval = int(auc_log_perc * len(train_loader))
    loss_log_interval = int(loss_log_perc * len(train_loader))

    run = neptune.init_run(
        project="oren.matar/BlenderShaders",
        api_token=NEPTUNE_KEY,
    )
    run["parameters"] = {
        "sample_perc": sample_perc,
        "learning_rate": learning_rate,
        "layers_taken": layers_to_take[0],
        "model_name": model_name,
        "final_emb": final_emb,
        "pool_type": pool_type,
        "loss_type": loss_type,
        "resize": resize,
        "scheduler_name": scheduler_name,
    }
    t = time.time()

    criterion = torch.nn.CosineEmbeddingLoss(margin=0.3) if loss_type == "cos" else ContrastiveLoss(margin=0.3)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    else:
        raise NotImplemented(f'{scheduler_name} is not a known scheduler name')

    # Training Loop
    for epoch in range(1, num_epochs+1):
        model.train()  # Set to training mode
        running_loss = 0.0

        for batch_idx, (img1, img2, labels, attributes) in enumerate(train_loader, start=1):
            if batch_idx % 200 == 0:
                logging.info(f"epoch {epoch}/{num_epochs}, batch: {batch_idx}/{len(train_loader)}. Time: {round(time.time() - t, 2)}")
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            embedding1 = model(img1)
            embedding2 = model(img2)

            targets = 2 * labels - 1  # Convert 0/1 labels to -1/1
            loss = criterion(embedding1, embedding2, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % auc_log_interval == 0:
                log_gpu_memory()
                eval_auc = evaluate_model_by_attribute(model, test_loader, criterion)
                run["test"].append(eval_auc)

            if batch_idx % loss_log_interval == 0:
                run["train/loss"].append(running_loss)
                running_loss = 0.0


        logging.info("Epoch done, saving")
        file_name = generate_checkpoint_name(
            models_path,
            epoch,
            layers_to_take=layers_to_take,
            learning_rate=learning_rate,
            model_name=model_name,
            final_emb=final_emb,
            scheduler_name=scheduler_name,
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            file_name,
        )

        scheduler.step()

    eval_auc = evaluate_model_by_attribute(model, test_loader, criterion)
    run["test"].append(eval_auc)
    run["file_name"] = file_name
    run["run_time"] = time.time() - t
    run.stop()


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Train the model with different hyperparameters.")

    # Define arguments to parse
    parser.add_argument(
        "--sample_perc", type=float, required=True, help="Percentage of dataset to use for training (0 to 1)."
    )
    parser.add_argument("--layers_to_take", type=str, required=True, help="List of layers to take from the model.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["vgg", "resnet", "dists"],
        help="Name of the model (used for saving).",
    )
    parser.add_argument("--final_emb", type=int, required=True, help="Size of the final embedding layer.")
    parser.add_argument("--scheduler_name", type=str, choices=["cosine", "step"], required=True, help="scheduler name")
    parser.add_argument(
        "--pool_type", type=str, choices=["avg", "max"], required=True, help="Pooling type ('avg' or 'max')."
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["cos", "cont"],
        required=True,
        help="Loss function type (e.g., 'contrastive', 'triplet').",
    )
    parser.add_argument("--resize", type=str, default=True, help="Whether to resize images (True or False).")
    parser.add_argument(
        "--test_perc", type=float, default=0.1, help="Percentage of dataset for testing (default: 0.1)."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs to train (default: 4).")

    # Parse the arguments
    args = parser.parse_args()

    set_seed()
    # Call the train function with the parsed arguments
    train(
        sample_perc=args.sample_perc,
        layers_to_take=[int(x) for x in args.layers_to_take.split("_")],  # Convert the str to a list
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        final_emb=args.final_emb,
        pool_type=args.pool_type,
        scheduler_name=args.scheduler_name,
        loss_type=args.loss_type,
        resize=args.resize == "True",
        test_perc=args.test_perc,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
