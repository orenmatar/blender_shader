import time

import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
# Import AMP scaler
from torch.cuda.amp import autocast, GradScaler
import gc # For garbage collection

from transformers import AutoTokenizer

from Logic.NN_models.image_embedders import load_resnet_model
from Logic.NN_models.images_to_code_model import ImageToCodeDecoder, SpecialTokenTokenizerWrapper
from Logic.NN_models.nn_models_utils import count_parameters

neptune_key = os.getenv("NEPTUNE_API_TOKEN")

class ImageCodeDataset(Dataset):
    def __init__(self, image_dir, codes_filepath, tokenizer, random_flips=True, max_seq_len=512):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        transformations = [transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224))]
        if random_flips:
            transformations.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
        transformations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45], std=[0.225]),
            ]
        )
        self.transform = transforms.Compose(transformations)
        self.max_seq_len = max_seq_len

        with open(codes_filepath, 'r') as f:
            self.codes_data = json.load(f)

        self.image_ids = list(self.codes_data.keys())

    def __len__(self):
        return len(self.image_ids)

    def open_image(self, image_filename):
        image = Image.open(image_filename)
        image = self.transform(image)
        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = os.path.join(self.image_dir, f"{image_id}.png")
        code_string = self.codes_data[image_id]

        # Load image
        image = self.open_image(image_filename)

        # Tokenize code string using your tokenizer's encode method
        # This will include SOS and EOS tokens.
        encoded_tokens = self.tokenizer.encode(code_string, add_special_tokens=True)
        encoded_tokens = encoded_tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(encoded_tokens))

        # Ensure the length is exactly max_seq_len (due to truncation logic)
        assert len(encoded_tokens) == self.max_seq_len, f"Tokenized sequence length mismatch: {len(encoded_tokens)} vs {self.max_seq_len}"


        # Create decoder input (shifted right) and target sequence
        # decoder_input_tokens: SOS, token1, token2, ..., tokenN (length = max_seq_len - 1)
        # target_tokens: token1, token2, ..., tokenN, EOS (length = max_seq_len - 1)
        # We need to ensure that the input to the decoder always has a sequence length of max_seq_len-1 (or less during generation)
        # and the output target has max_seq_len-1 as well.
        # This means the `encoded_tokens` must be of length `max_seq_len`.
        # The first token (SOS) is for the decoder input, the last token (EOS or PAD) is for the target.

        decoder_input_tokens = torch.tensor(encoded_tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(encoded_tokens[1:], dtype=torch.long)

        # Convert boolean padding mask to a float mask with -inf for padded positions
        tgt_key_padding_mask = torch.full_like(decoder_input_tokens, 0.0, dtype=torch.float)
        tgt_key_padding_mask.masked_fill_(decoder_input_tokens == self.tokenizer.pad_token_id, float('-inf'))

        return image, decoder_input_tokens, target_tokens, tgt_key_padding_mask


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, neptune_run,
                model_save_path="best_model.pth", log_interval=100,
                gradient_accumulation_steps=1):
    model.to(device)
    best_val_loss = float('inf')
    best_model_save_path = model_save_path  # Correctly initialized here
    scaler = GradScaler()

    # Initialize the learning rate scheduler
    # We'll reduce LR by a factor of 0.1 if validation loss doesn't improve for 3 epochs (threshold 1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=1e-2)

    # Optional: Compile the model (PyTorch 2.0+)
    # model = torch.compile(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        epoch_start_time = time.time()
        num_batches = len(train_loader)

        for batch_idx, (images, decoder_input_tokens, target_tokens, tgt_padding_mask) in enumerate(train_loader):
            images = images.to(device)
            decoder_input_tokens = decoder_input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)

            with autocast():
                logits = model(images, decoder_input_tokens, tgt_padding_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), target_tokens.reshape(-1))
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

            if batch_idx % log_interval == 0:
                current_loss = loss.item() * gradient_accumulation_steps
                time_elapsed = time.time() - epoch_start_time
                batches_processed = batch_idx + 1
                avg_time_per_batch = time_elapsed / batches_processed
                time_remaining_seconds = avg_time_per_batch * (num_batches - batches_processed)
                time_elapsed_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time_elapsed))
                time_remaining_str = time.strftime('%Hh %Mm %Ss', time.gmtime(time_remaining_seconds))

                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{num_batches}], "
                      f"Loss: {current_loss:.4f} (Effective Batch Size: {train_loader.batch_size * gradient_accumulation_steps}), "
                      f"Time Elapsed: {time_elapsed_str}, Est. Time Rem.: {time_remaining_str}")

                if neptune_run:
                    neptune_run[f"train/batch_loss"].append(current_loss)
                    neptune_run["learning_rate"].append(optimizer.param_groups[0]["lr"])

            if batch_idx % (log_interval * 5) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Handle any remaining accumulated gradients at the end of the epoch
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}")
        if neptune_run:
            neptune_run[f"train/epoch_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_total_loss = 0
        val_start_time = time.time()
        with torch.no_grad():
            for batch_idx, (images, decoder_input_tokens, target_tokens, tgt_padding_mask) in enumerate(val_loader):
                images = images.to(device)
                decoder_input_tokens = decoder_input_tokens.to(device)
                target_tokens = target_tokens.to(device)
                tgt_padding_mask = tgt_padding_mask.to(device)

                with autocast():
                    logits = model(images, decoder_input_tokens, tgt_padding_mask)
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), target_tokens.reshape(-1))
                val_total_loss += loss.item()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        avg_val_loss = val_total_loss / len(val_loader)
        val_duration = time.time() - val_start_time
        val_duration_str = time.strftime('%Hh %Mm %Ss', time.gmtime(val_duration))

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Average Validation Loss: {avg_val_loss:.4f}, Validation Duration: {val_duration_str}")
        if neptune_run:
            neptune_run[f"val/epoch_loss"].append(avg_val_loss)
            neptune_run[f"val/duration_seconds"].append(val_duration)

        # --- Scheduler Step ---
        # The scheduler steps based on the validation loss
        scheduler.step(avg_val_loss)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Learning rate after scheduler step: {optimizer.param_groups[0]['lr']:.6f}")
        # Log learning rate to Neptune after scheduler step
        if neptune_run:  # Ensure neptune_run exists before logging
            neptune_run["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # --- Save model for current epoch and log to Neptune ---
        # Construct a unique path for the current epoch's model
        epoch_model_path = os.path.join(os.path.dirname(model_save_path),
                                        f"model_epoch_{epoch + 1:03d}.pth")
        model.save_state_dict(epoch_model_path)
        print(f"Saved model for Epoch {epoch + 1} to {epoch_model_path}")
        if neptune_run:
            # Upload this specific epoch's model to Neptune
            neptune_run[f"model_checkpoints/epoch_{epoch + 1:03d}"].upload(epoch_model_path)
            print(f"Epoch {epoch + 1} model uploaded to Neptune.")

        # --- Best Model Saving ---
        # This logic determines if the current model is the best so far based on validation loss
        # and saves it to a designated "best model" path. Training continues regardless.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Save the model to the 'best_model_save_path'
            model.save_state_dict(best_model_save_path)
            print(f"Saved BEST model to {best_model_save_path} with validation loss: {best_val_loss:.4f}")
            if neptune_run:
                # Upload the 'best_model.pth' to a specific key in Neptune
                neptune_run["model_checkpoints/best_model_state_dict"].upload(best_model_save_path)
                neptune_run["metrics/best_val_loss"] = best_val_loss  # Log the best validation loss
                print("Best model state_dict uploaded to Neptune.")

    print("Training finished.")

if __name__ == '__main__':
    BATCH_SIZE = 8
    NUM_EPOCHS = 24
    LEARNING_RATE = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 4

    # Configuration
    IMAGE_DIR = '/Users/orenm/BlenderShaderProject/data/images/'
    CODES_FILE = '/Users/orenm/BlenderShaderProject/data/datasets/network_managers_strings.json'
    ACTIVE_MODELS_PATH = '/Users/orenm/BlenderShaderProject/data/active_models/'
    image_embedder_for_texture = 'ep_4_la_7_256_le_0_0001_mo_resnet_fi_128_sc_cosine.pt'
    TOKENIZER_PATH = os.path.join(ACTIVE_MODELS_PATH, "my_tokenizer")
    image_embedder_for_texture_path = os.path.join(ACTIVE_MODELS_PATH, image_embedder_for_texture)

    MODEL_SAVE_PATH = os.path.join(ACTIVE_MODELS_PATH, "image_to_code_decoder_final.pth")
    IMAGE_EMBEDDER_OUTPUT_DIM = 128  # Must match your embedder's actual output dimension

    run = neptune.init_run(
        project="oren.matar/BlenderShaders",
        api_token=neptune_key,
    )
    hyperparameters = {
        "image_dir": IMAGE_DIR,
        "codes_file": CODES_FILE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "image_embedder_output_dim": IMAGE_EMBEDDER_OUTPUT_DIM,
        "criterion": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "gradient_clipping_norm": 1.0,
        "amp_enabled": True
    }
    run["hyperparameters"] = hyperparameters

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer (replace with your actual tokenizer)
    print("Initializing Tokenizer...")
    # Add special tokens
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer = SpecialTokenTokenizerWrapper(tokenizer)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    strings = json.load(open(CODES_FILE, "r"))
    lengths = [len(tokenizer.encode(x)) for x in strings.values()]
    MAX_SEQUENCE_LENGTH = max(lengths) + 15
    print(f'Max seq length: {MAX_SEQUENCE_LENGTH}')

    # 2. Initialize Image Embedder (replace with your actual embedder)
    print("Initializing Image Embedder...")

    image_embedder = load_resnet_model(image_embedder_for_texture_path)

    # 3. Initialize Model
    print("Initializing ImageToCodeDecoder Model...")
    model = ImageToCodeDecoder(
        embedder=image_embedder,
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        image_emb_dim=IMAGE_EMBEDDER_OUTPUT_DIM,
        model_dim=768,
        num_layers=16,
        num_heads=12,
        max_seq_len=MAX_SEQUENCE_LENGTH,  # Pass MAX_SEQUENCE_LENGTH here
        pad_token_id=tokenizer.pad_token_id,
        sos_token_id=tokenizer.sos_token_id,
        eos_token_id=tokenizer.eos_token_id
    ).to(device)

    total_trainable_params = count_parameters(model)
    print(f"Model initialized with {total_trainable_params:,} trainable parameters.")

    # 5. Create Datasets and DataLoaders
    print("Creating Datasets and DataLoaders...")
    dataset = ImageCodeDataset(
        image_dir=IMAGE_DIR,
        codes_filepath=CODES_FILE,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQUENCE_LENGTH  # Pass MAX_SEQUENCE_LENGTH to the dataset
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # 6. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 7. Start Training
    print("Starting training...")
    train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device, run, MODEL_SAVE_PATH,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
