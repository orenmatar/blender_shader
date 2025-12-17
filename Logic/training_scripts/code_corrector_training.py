import os
import json
import time
import neptune
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

from torch import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
import torch.utils.checkpoint as checkpoint
from collections import defaultdict
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef

from Logic.constants import (
    NO_ACTION_ID,
    MAIN_HEAD_LABEL_TO_ID_MAP,
    SECONDARY_HEAD_LABEL_TO_ID_MAP,
    SECONDARY_HEAD_ID_TO_LABEL_MAP,
    SECONDARY_HEAD_WEIGHTS,
)
from Logic.NN_models.data_loaders import (
    CodeCorrectorDataset,
    CustomDataCollatorForTokenClassification,
)
from Logic.NN_models.code_corrector_model import CodeCorrector
from Logic.NN_models.nn_models_utils import count_parameters
from Logic.training_scripts.scripts_utils import (
    generate_checkpoint_name,
    log_gpu_memory,
    set_seed,
    upload_to_gcp,
)

# this is required for torch compile to work without triton, but not using compile now because it messes up checkpoints
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

neptune_key = os.getenv("NEPTUNE_API_TOKEN")


def evaluate_model(model, data_loader, batch_size, loss_func):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting eval using {device}")

    t = time.time()
    model.eval()
    all_data = defaultdict(list)
    running_loss = 0
    all_labels_predicted1 = []
    all_labels_predicted2 = []
    running_mmc1 = []
    running_mmc2 = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i % 50 == 0:
                print(f"Test batch: {i}/{len(data_loader)}. Time: {round(time.time() - t, 2)}")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_imgs = batch["target_imgs"].to(device)
            source_imgs = batch["source_imgs"].to(device)
            add_cur_images = batch["add_cur_images"].to(device)
            labels_head1 = batch["labels_head1"].to(device)
            labels_head2 = batch["labels_head2"].to(device)
            distances = batch["distances"].to(device)

            match_value, main_token_cls, secondary_token_cls = model(
                input_ids, attention_mask, target_imgs, source_imgs, add_cur_images
            )

            loss, seq_no_action = loss_func(
                match_value, main_token_cls, secondary_token_cls, distances, labels_head1, labels_head2
            )
            running_loss += loss.item()

            predicted_head1 = main_token_cls.argmax(dim=2).detach().cpu().numpy()
            predicted_head2 = secondary_token_cls.argmax(dim=2).detach().cpu().numpy()
            labels1 = labels_head1.detach().cpu().numpy()
            labels2 = labels_head2.detach().cpu().numpy()

            accuracy1 = (predicted_head1 == labels1).mean(axis=1, where=labels1 != -100)
            accuracy2 = (predicted_head2 == labels2).mean(axis=1, where=labels2 != -100)
            accuracy1_non0 = np.nanmean(predicted_head1 == labels1, axis=1, where=(~np.isin(labels1, [-100, 0])))
            accuracy2_non0 = np.nanmean(predicted_head2 == labels2, axis=1, where=(~np.isin(labels2, [-100, 0])))

            mcc1 = matthews_corrcoef(predicted_head1.flatten(), labels1.flatten())
            mcc2 = matthews_corrcoef(predicted_head2.flatten(), labels2.flatten())

            all_data["with_image"].extend(add_cur_images.detach().cpu().numpy())
            all_data["distances"].extend(distances.detach().cpu().numpy())
            all_data["distance_score"].extend(match_value.flatten().detach().cpu().numpy())
            all_data["seq_no_action"].extend(seq_no_action.flatten().detach().cpu().numpy())

            all_data["accuracy1"].extend(accuracy1)
            all_data["accuracy2"].extend(accuracy2)
            all_data["accuracy1_non0"].extend(accuracy1_non0)
            all_data["accuracy2_non0"].extend(accuracy2_non0)

            all_labels_predicted1.extend(predicted_head1.flatten())
            all_labels_predicted2.extend(predicted_head2.flatten())

            running_mmc1.append(mcc1)
            running_mmc2.append(mcc2)

    all_data_df = pd.DataFrame(all_data)
    # count labels that appear with some frequency
    all_labels_predicted1 = pd.Series(all_labels_predicted1)
    all_labels_predicted1 = all_labels_predicted1[all_labels_predicted1 != 0]
    all_labels_predicted1 = all_labels_predicted1.value_counts(normalize=True)
    all_labels_predicted1 = all_labels_predicted1[all_labels_predicted1 > 0.005]
    all_labels_predicted2 = pd.Series(all_labels_predicted2)
    all_labels_predicted2 = all_labels_predicted2[all_labels_predicted2 != 0]
    all_labels_predicted2 = all_labels_predicted2.value_counts(normalize=True)
    all_labels_predicted2 = all_labels_predicted2[all_labels_predicted2 > 0.005]

    metrics = {
        "running_loss": running_loss / len(data_loader) * batch_size,
        "mmc_head1": np.mean(running_mmc1),
        "mmc_head2": np.mean(running_mmc2),
    }

    def make_metrics_from_df(df):
        res = {
            "accuracy_head_1": df["accuracy1"].mean(),
            "accuracy_head_2": df["accuracy2"].mean(),
            "accuracy_no0_head_1": df["accuracy1_non0"].mean(),
            "accuracy_no0_head_2": df["accuracy2_non0"].mean(),
            "no_action_prec": df["seq_no_action"].mean(),
            "distance_score_all": -pearsonr(df["distances"], df["distance_score"]).statistic,
            "n_predicted1": len(all_labels_predicted1),
            "n_predicted2": len(all_labels_predicted2),
        }
        close_to_target = df[df["distances"] < 6]
        res["distance_score_close"] = -pearsonr(
            close_to_target["distances"], close_to_target["distance_score"]
        ).statistic
        res["accuracy1_distance_corr"] = pearsonr(close_to_target["accuracy1"], close_to_target["distances"]).statistic
        return res

    metrics.update(
        {f"{key}_with_image": value for key, value in make_metrics_from_df(all_data_df[all_data_df.with_image]).items()}
    )
    metrics.update(
        {f"{key}_no_image": value for key, value in make_metrics_from_df(all_data_df[~all_data_df.with_image]).items()}
    )

    print(f"Eval took: {round(time.time() - t,2)}")

    model.train()
    return metrics


def train(
    learning_rate: float,
    path: str,
    batch_size: int,
    bert_size: str = "small",
    num_epochs: int = 2,
    token_loss_weight=1,
    no_action_penalty=2,
    run_number=0,
    limit_to_distance=None,
    accumulation_steps=4,
    log_test_eval_every=0.35,
    log_loss_every=0.005,
    weight_decay=0.001,
    seed=40,
    full_model_file_name=None,
    image_embedder_file_name=None,
    code_embedder_file_name=None,
    neptune_run_id=None,
    upload_files=True,
):
    set_seed(seed)
    persistent_workers = True
    num_workers = 4
    images_path = os.path.join(path, "images/")
    dataset_path = os.path.join(path, "datasets/")
    active_models_path = os.path.join(path, "active_models/")
    all_models_path = os.path.join(path, "models/")
    tokenizer_path = os.path.join(active_models_path, "my_tokenizer")

    train_dataset_file = os.path.join(dataset_path, "train_dataset_for_corrector.json")
    test_dataset_file = os.path.join(dataset_path, "test_dataset_for_corrector.json")
    with open(train_dataset_file, "r") as f:
        train_data = json.load(f)
    with open(test_dataset_file, "r") as f:
        test_data = json.load(f)

    print(f"Loaded {len(train_data)} train examples")
    if limit_to_distance:
        train_data = [x for x in train_data if x[3] <= limit_to_distance]
        print(f"Limited to distance of {limit_to_distance}, using {len(train_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_dataset = CodeCorrectorDataset(train_data, images_path)
    test_dataset = CodeCorrectorDataset(test_data, images_path)
    collator = CustomDataCollatorForTokenClassification(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    test_eval_interval = int(log_test_eval_every * len(train_dataloader))
    loss_log_interval = max(1, int(log_loss_every * len(train_dataloader)))
    print(f"Dataset size: {len(train_data)}. Eval every {test_eval_interval}. Loss every {loss_log_interval}")
    print("Loading models")

    main_head_n_tokens = len(MAIN_HEAD_LABEL_TO_ID_MAP)
    secondary_head_n_tokens = len(SECONDARY_HEAD_LABEL_TO_ID_MAP)
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

    def full_loss(corr_value, tokens_predictions1, tokens_predictions2, true_distances, labels1, labels2):
        distance_loss = distance_loss_criterion(corr_value, (1 - true_distances / 20).reshape(-1, 1))
        main_head_loss = main_head_criterion(
            tokens_predictions1.permute(0, 2, 1), labels1
        )  # convert batch, sequence, labels to batch, labels, sequence
        secondary_head_loss = secondary_head_criterion(tokens_predictions2.permute(0, 2, 1), labels2)

        # check if it predicted any action
        predicted_no_action = torch.argmax(tokens_predictions1, dim=2) == NO_ACTION_ID
        # check if predicted no action only where the labels are important, not -100
        predicted_no_action_on_labels = predicted_no_action & (labels1 != -100)
        # only for example with distance !=0, so ones where an action needs to be taken...
        no_action_sequences = predicted_no_action_on_labels.all(dim=1) * (true_distances != 0)
        n_seq_no_action = no_action_sequences.sum()
        no_action_loss = n_seq_no_action * no_action_penalty / batch_size

        loss = (
            distance_loss
            + token_loss_weight * main_head_loss
            + token_loss_weight * secondary_head_loss
            + no_action_loss
        )
        return loss, no_action_sequences

    if full_model_file_name is not None:
        model_path = os.path.join(active_models_path, full_model_file_name)
        model = CodeCorrector.load(model_path)
    else:
        image_emb_path = os.path.join(active_models_path, image_embedder_file_name)
        model = CodeCorrector.new_model(
            tokenizer_path, image_emb_path, main_head_n_tokens, secondary_head_n_tokens, image_token=154
        )
    model.to(device)
    print(f"Model has {count_parameters(model)} params")

    run = neptune.init_run(
        project="oren.matar/BlenderShaders",
        api_token=neptune_key,
        with_id=neptune_run_id,
    )
    run["parameters"] = {
        "learning_rate": learning_rate,
        "bert_size": bert_size,
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_training_steps = int(num_epochs * len(train_dataloader) / accumulation_steps)
    warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=3,
    )

    def save():
        print("Saving...")
        model_path_name = generate_checkpoint_name(
            all_models_path,
            epoch,
            model_name="code_corrector",
            bert_size=bert_size,
            run_number=run_number,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
        )
        torch.save(
            model.make_state_dict(),
            model_path_name,
        )
        model_file_name = model_path_name.split("/")[-1]
        if upload_files:
            try:
                upload_to_gcp("blender-shader", model_path_name, f"models/{model_file_name}")
            except:
                print("Upload FAILED!!! Continuing with the hope it will fix itself")
        return model_file_name

    scaler = GradScaler()
    t = time.time()

    # Training Loop
    print("Starting training")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader, start=1):
            if batch_idx % 500 == 0:
                print(
                    f"epoch {epoch}/{num_epochs}, batch: {batch_idx}/{len(train_dataloader)}. Time: {round(time.time() - t, 2)}"
                )

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_imgs = batch["target_imgs"].to(device)
            source_imgs = batch["source_imgs"].to(device)
            add_cur_images = batch["add_cur_images"].to(device)
            labels_head1 = batch["labels_head1"].to(device)
            labels_head2 = batch["labels_head2"].to(device)
            distances = batch["distances"].to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                match_value, main_token_cls, secondary_token_cls = checkpoint.checkpoint(
                    model, input_ids, attention_mask, target_imgs, source_imgs, add_cur_images, use_reentrant=False
                )
                loss, no_action_sequences = full_loss(
                    match_value, main_token_cls, secondary_token_cls, distances, labels_head1, labels_head2
                )

            scaler.scale(loss).backward()

            if batch_idx % accumulation_steps == 0:
                scaler.unscale_(optimizer)  # Unscales gradients before clipping
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            running_loss += loss.item()

            if batch_idx % test_eval_interval == 0:
                log_gpu_memory()
                eval_values = evaluate_model(model, test_dataloader, batch_size, full_loss)
                run["test"].append(eval_values)
                run["token_loss_weight"].append(token_loss_weight)
                run["no_action_penalty"].append(no_action_penalty)
                run["limit_to_distance"].append(limit_to_distance if limit_to_distance else 0)
                file_name = save()
                run["file_names"].append(file_name)

            if batch_idx % loss_log_interval == 0:
                run["train/loss"].append(running_loss / (loss_log_interval * batch_size))
                run["train/learning_rate"].append(scheduler.get_lr()[0])
                running_loss = 0.0

        eval_values = evaluate_model(model, test_dataloader, batch_size, full_loss)
        run["test"].append(eval_values)
        run["token_loss_weight"].append(token_loss_weight)
        run["no_action_penalty"].append(no_action_penalty)
        run["limit_to_distance"].append(limit_to_distance if limit_to_distance else 0)
        print("Epoch done, saving")
        file_name = save()
        run["file_names"].append(file_name)

    run["run_time"] = time.time() - t
    run.stop()
