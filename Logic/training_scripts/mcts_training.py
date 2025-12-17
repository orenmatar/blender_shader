import datetime
import gc
import logging
import os
import random
import re
import neptune
import torch
import warnings

from torch import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
import torch.utils.checkpoint as checkpoint
from collections import deque
import torch.nn as nn

from Logic.bpy_connector import _reset_blender
from Logic.constants import (
    MAIN_HEAD_LABEL_TO_ID_MAP,
    SECONDARY_HEAD_ID_TO_LABEL_MAP,
    SECONDARY_HEAD_WEIGHTS,
)
from Logic.NN_models.data_loaders import CustomDataCollatorForMCTSTraining, MCTSDataset
from Logic.mcts_operator import MCTSOperator, search_metrics
from Logic.training_scripts.mcts_training_utils import filter_nodes_by_distance, node_to_mcts_training_example
from Logic.training_scripts.scripts_utils import (
    log_gpu_memory,
    set_seed,
    upload_to_gcp,
)

# this is required for torch compile to work without triton ¯\_(ツ)_/¯, but not using compile now because it messes up checkpoints
import torch._dynamo

from Logic.utils import create_unique_subdir

torch._dynamo.config.suppress_errors = True

neptune_key = os.getenv("NEPTUNE_API_TOKEN")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


def get_most_recent_corrector_file(directory):
    pattern = re.compile(r"^corrector_mcts_(\d{8}_\d{6})\.pt$")

    latest_file = None
    latest_time = None

    for file_name in os.listdir(directory):
        match = pattern.match(file_name)
        if match:
            timestamp_str = match.group(1)
            try:
                file_time = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if latest_time is None or file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_name
            except ValueError:
                continue

    return latest_file


def train(
    learning_rate: float,
    path: str,
    batch_size: int,
    tokenizer_path: str,
    image_embedder_path: str,
    target_images_path: str,
    num_trees: int = 3000,
    reward_prediction_weight=0.8,
    max_expansions=25,
    max_nodes_to_expand_per_iter=10,
    search_method="sample",
    accumulation_steps=4,
    weight_decay=0.001,
    save_every=10,
    seed=40,
    tree_sample_size=100,
    neptune_run_id=None,
    upload_files=True,
):
    set_seed(seed)
    persistent_workers = True
    num_workers = 4
    mcts_workdir = os.path.join(path, "mcts_workdir/")
    active_models_path = os.path.join(path, "active_models/")

    # Set up logging configuration
    log_filename = os.path.join(path, "mcts_training_logger.log")
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

    logging.info("Starting a new run...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} device")

    model_name = get_most_recent_corrector_file(active_models_path)
    logging.info(f"Loading model from path: {model_name}")
    model_path = os.path.join(active_models_path, model_name)
    mcts_operator = MCTSOperator(model_path, tokenizer_path, image_embedder_path)
    collator = CustomDataCollatorForMCTSTraining(mcts_operator.tokenizer)

    main_head_n_tokens = len(MAIN_HEAD_LABEL_TO_ID_MAP)
    distance_loss_criterion = nn.MSELoss()

    # set the 0 label (no action) to have a lesser weight, so the model knows it's ok to guess other stuff
    # suggest as many actions as you want if they have some likelihood...
    main_head_weights = torch.tensor([0.06] + [1] * (main_head_n_tokens - 1)).to(device)
    # add more weights for secondary - because some classes appear way too much
    sorted_ids = sorted(SECONDARY_HEAD_ID_TO_LABEL_MAP)
    secondary_head_weights = [SECONDARY_HEAD_WEIGHTS.get(SECONDARY_HEAD_ID_TO_LABEL_MAP[i], 1) for i in sorted_ids]
    secondary_head_weights = torch.tensor(secondary_head_weights).to(device)
    main_head_criterion = nn.CrossEntropyLoss(weight=main_head_weights)
    secondary_head_criterion = nn.CrossEntropyLoss(weight=secondary_head_weights)
    model = mcts_operator.code_corrector
    replay_buffer = deque(maxlen=700)

    run = neptune.init_run(
        project="oren.matar/BlenderShaders",
        api_token=neptune_key,
        with_id=neptune_run_id,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    num_training_steps = int(num_trees * tree_sample_size / batch_size / accumulation_steps)
    warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=4,
    )

    def save():
        logging.info("Saving...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path_name = os.path.join(active_models_path, f"corrector_mcts_{timestamp}.pt")
        torch.save(
            model.make_state_dict(),
            model_path_name,
        )
        model_file_name = model_path_name.split("/")[-1]
        if upload_files:
            try:
                upload_to_gcp("blender-shader", model_path_name, f"models/{model_file_name}")
            except:
                logging.info("Upload FAILED!!! Continuing with the hope it will fix itself")
        return model_file_name

    scaler = GradScaler()
    image_file_names = os.listdir(target_images_path)
    total_examples_trained = 0
    # Training Loop
    for epoch in range(1, num_trees + 1):
        logging.info(f"Woring on tree {epoch}")
        model.eval()
        _reset_blender()
        gc.collect()
        torch.cuda.empty_cache()

        # Step 1: run the mcts
        current_dir = create_unique_subdir(mcts_workdir)
        selected_image = random.choice(image_file_names)
        target_img_path = os.path.join(target_images_path, selected_image)
        logging.info(f"Selected image: {selected_image}")

        try:
            graph_manager, all_node_expansions = mcts_operator.search(
                target_img_path,
                current_dir,
                max_expansions=max_expansions,
                max_nodes_to_expand_per_iter=max_nodes_to_expand_per_iter,
                search_method=search_method,
                c_puct=3,
                temperature=0.1,
                sample_labels=True,
                optimize=False,
                n_nodes_to_optimize_at_end=3,
            )
        except Exception as e:
            logging.error(f"[ERROR], failed: {e}", exc_info=True)
            continue

        logging.info(f"Finished building tree, extracting examples...")

        # Step 2: extract the learning examples
        legit_nodes = filter_nodes_by_distance(graph_manager.network, mcts_operator.start_node_id, min_depth=8)
        # only nodes that were expanded at least once - they have a texture, they have suggested actions from them
        legit_nodes = [n for n in legit_nodes if graph_manager.network.out_degree(n) > 0]
        sampled_nodes = random.sample(list(legit_nodes), k=min(tree_sample_size, len(legit_nodes)))
        logging.info(f"Sampled {len(sampled_nodes)} nodes from the tree")
        all_examples = []
        for node_id in sampled_nodes:
            node_training_data = node_to_mcts_training_example(
                graph_manager,
                node_id,
                current_dir,
                target_img_path,
                mcts_operator,
                add_cur_image=True,
                reward_prediction_weight=reward_prediction_weight,
            )
            all_examples.append(node_training_data)

        # take an equal amount from replay buffer
        past_examples = random.sample(replay_buffer, k=min(tree_sample_size, len(replay_buffer)))
        # take all new examples, and sample from past
        examples_for_training = list(all_examples) + past_examples
        replay_buffer.extend(all_examples)
        total_examples_trained += len(examples_for_training)
        logging.info(f"Training with {len(examples_for_training)} examples")

        # make dataset
        dataset = MCTSDataset(examples_for_training, random_flips=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        # Step 3: train on examples
        model.train()
        running_loss = 0
        logging.info(f"Starting training with examples")
        for batch_idx, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_imgs = batch["target_imgs"].to(device)
            source_imgs = batch["source_imgs"].to(device)
            add_cur_images = batch["add_cur_images"].to(device)
            labels_head1 = batch["labels_head1"].to(device)
            labels_head2 = batch["labels_head2"].to(device)
            target_values = batch["target_values"].to(device).float()

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                match_value, main_token_cls, secondary_token_cls = checkpoint.checkpoint(
                    model, input_ids, attention_mask, target_imgs, source_imgs, add_cur_images, use_reentrant=False
                )
                main_head_loss = main_head_criterion(main_token_cls.permute(0, 2, 1), labels_head1)
                secondary_head_loss = secondary_head_criterion(secondary_token_cls.permute(0, 2, 1), labels_head2)
                match_prediction_loss = distance_loss_criterion(match_value, target_values.view(-1, 1))
                loss = main_head_loss + secondary_head_loss + match_prediction_loss

            scaler.scale(loss).backward()

            if batch_idx % accumulation_steps == 0:
                scaler.unscale_(optimizer)  # Unscales gradients before clipping
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            running_loss += loss.item()

        logging.info(f"Finished training. So far: {total_examples_trained} examples")
        # evaluate results
        logging.info(f"Evaluating results")
        res = search_metrics(graph_manager, all_node_expansions, mcts_operator.start_node_id)
        log_gpu_memory()
        run["learning_rate"].append(scheduler.get_lr()[0])
        run["best_score"].append(res["best_score"])
        run["best_distance"].append(res["best_distance"])
        run["best_step"].append(res["best_step"])
        run["running_loss"].append(running_loss / len(examples_for_training))

        if epoch % save_every == 0:
            file_name = save()
            run["file_names"].append(file_name)

    run.stop()
