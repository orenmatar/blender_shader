import logging
import os
import time
from collections import defaultdict
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image


class SiameseDataset(Dataset):
    def __init__(self, data_pairs: list[tuple[str, str, str]], images_folder: str, resize=True, with_names=False):
        """
        Used to create a dataset for siamese network training - teaching a model to compare textures and determine
        whether they are similar or not.
        Args:
            data_pairs: Each tuple is (image1_path, image2_path, label - are they similar or not).
        """
        transformations = []
        if resize:
            transformations.append(transforms.Resize((224, 224)))
        transformations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45], std=[0.225]),
            ]
        )
        self.transform_grayscale = transforms.Compose(transformations)
        self.with_names = with_names
        self.data_pairs = data_pairs
        self.images_folder = images_folder

    def load_image(self, image_path):
        """Loads and preprocesses a single image."""
        full_path = f"{self.images_folder}/{image_path}.png"
        image = Image.open(full_path)
        return self.transform_grayscale(image)

    def __getitem__(self, idx):
        """Fetch a single data point."""
        img1_path, img2_path, label, attribute = self.data_pairs[idx]
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)
        if not self.with_names:
            return img1, img2, torch.tensor(label, dtype=torch.float32), attribute
        return img1, img2, torch.tensor(label, dtype=torch.float32), attribute, (img1_path, img2_path)

    def __len__(self):
        return len(self.data_pairs)


def create_siam_images_dataloaders(
    image_pairs, images_path, test_size=0.2, batch_size=32, num_workers=4, persistent_workers=True, **kwargs
):
    """
    Splits the dataset into train and test and creates DataLoader instances.
    """
    train_data, test_data = train_test_split(
        image_pairs, test_size=test_size, stratify=[attribute for _, _, label, attribute in image_pairs]
    )

    train_dataset = SiameseDataset(train_data, images_path, **kwargs)
    test_dataset = SiameseDataset(test_data, images_path, **kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    return train_loader, test_loader


def evaluate_model_by_attribute_image_emb(model: nn.Module, test_loader: DataLoader, criterion: nn.Module):
    """
    Evaluates the model on the test dataset and computes AUC scores for different attributes (textures that are very close
    distance texture, completely unrelated...).
    """
    all_scores = defaultdict(list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting eval using {device}")

    model.eval()  # Set the model to evaluation mode
    t = time.time()
    # Iterate over the test data
    running_loss = 0
    with torch.no_grad():
        for i, (img1, img2, labels, attributes) in enumerate(test_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            embedding1 = model(img1)
            embedding2 = model(img2)

            targets = 2 * labels - 1  # Convert 0/1 labels to -1/1
            loss = criterion(embedding1, embedding2, targets)
            running_loss += loss.item()

            # Calculate similarity (e.g., cosine similarity)
            similarity = F.cosine_similarity(embedding1, embedding2)

            # Append the similarity score and label to corresponding lists
            for idx, attr in enumerate(attributes):
                if attr == "similar_pairs":
                    all_scores["similar_pairs"].append(similarity[idx].cpu().numpy())
                else:
                    all_scores[attr].append(similarity[idx].cpu().numpy())

    auc_scores = {"running_loss": running_loss}
    true_scores = all_scores["similar_pairs"]
    true_labels = [1] * len(true_scores)
    for key in [key for key in all_scores if key != "similar_pairs"]:
        false_scores = all_scores[key]
        false_labels = [0] * len(false_scores)
        auc_scores[key] = roc_auc_score(false_labels + true_labels, false_scores + true_scores)

    logging.info(f"Eval took: {round(time.time() - t,2)}")
    model.train()
    return auc_scores


class CodeImageDataset(Dataset):
    """
    Dataset for loading images and corresponding code strings, to teach a model to associate images with code.
    """
    def __init__(self, dataset, code_strings: dict[str, str], image_folder: str):
        transformations = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45], std=[0.225]),
        ]
        self.code_strings = code_strings
        self.dataset = dataset
        self.image_folder = image_folder
        self.transform = transforms.Compose(transformations)
        self.label_margin_map = {
            "MATCHING": 0.5,
            "RAND": 0.3,
            "DISTANT_IN_CLUSTER": 0.5,
            "CLOSE_IN_CLUSTER": 0.65,
            "ONE_AWAY_STURTUCAL": 0.8,
            "ONE_AWAY_PARAM": 0.85,
        }
        self.label_map = {
            "MATCHING": 1,
            "RAND": 0,
            "DISTANT_IN_CLUSTER": 0,
            "CLOSE_IN_CLUSTER": 0,
            "ONE_AWAY_STURTUCAL": 0,
            "ONE_AWAY_PARAM": 0,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name, code_key, label_str = self.dataset[idx]
        img_path = os.path.join(self.image_folder, f"{img_name}.png")
        image = Image.open(img_path)
        image = self.transform(image)

        code_str = self.code_strings[code_key]
        label = self.label_map[label_str]
        label_margin = self.label_margin_map[label_str]

        return image, code_str, label, label_str, label_margin


class CodeCorrectorDataset(Dataset):
    """
    Dataset for loading images and corresponding code strings, and the edits, or corrections, that need to be made,
    to teach a model to correct code based on images.
    """
    def __init__(self, dataset, image_folder, random_flips=True):
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
        self.dataset = dataset
        self.image_folder = image_folder
        self.transform = transforms.Compose(transformations)

    def __len__(self):
        return len(self.dataset)

    def open_image(self, img_name=None, img_path=None):
        if img_name is not None:
            img_path = os.path.join(self.image_folder, f"{img_name}.png")
        assert img_path is not None, "Either img_name or img_path must be provided"
        image = Image.open(img_path)
        return self.transform(image)

    def __getitem__(self, idx):
        from_node, target_node, add_cur_image, distance, example_data = self.dataset[idx]
        target_img = self.open_image(target_node)
        source_img = target_img
        if add_cur_image:
            source_img = self.open_image(from_node)
        input_ids = example_data["input_ids"]
        attention = torch.ones(len(input_ids))

        return (
            target_img,
            source_img,
            add_cur_image,
            distance,
            input_ids,
            attention,
            example_data["main_head_ids"],
            example_data["secondary_head_ids"],
        )


class CustomDataCollatorForTokenClassification:
    def __init__(self, tokenizer, loss_padding_value=-100):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer(tokenizer.special_tokens_map["pad_token"])["input_ids"][
            0
        ]  # yes I know it's stupid there was a problem with getting it directly
        self.loss_padding_value = loss_padding_value

    def __call__(self, batch):
        # Find the longest sequence length in the batch
        max_length = max(len(input_ids) for _, _, _, _, input_ids, _, _, _ in batch)

        # Initialize lists to store padded values
        input_ids = []
        attention_mask = []
        labels_head1 = []
        labels_head2 = []
        target_imgs = []
        source_imgs = []
        distances = []
        add_cur_images = []

        # Iterate over each sample in the batch
        for (
            target_img,
            source_img,
            add_cur_image,
            distance,
            input_id,
            attention,
            main_head_ids,
            secondary_head_ids,
        ) in batch:
            # Pad input_ids and attention_mask using tensor operations
            input_ids.append(self.pad_sequence(input_id, max_length, pad_value=self.pad_token_id))
            attention_mask.append(self.pad_sequence(attention, max_length, pad_value=0))

            # Pad labels (both heads) using tensor operations
            labels_head1.append(self.pad_sequence(main_head_ids, max_length, pad_value=self.loss_padding_value))
            labels_head2.append(self.pad_sequence(secondary_head_ids, max_length, pad_value=self.loss_padding_value))
            target_imgs.append(target_img)
            source_imgs.append(source_img)
            distances.append(distance)
            add_cur_images.append(add_cur_image)

        # Convert lists to tensors
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels_head1 = torch.stack(labels_head1)
        labels_head2 = torch.stack(labels_head2)
        target_imgs = torch.stack(target_imgs)
        source_imgs = torch.stack(source_imgs)
        distances = torch.tensor(distances)
        add_cur_images = torch.tensor(add_cur_images)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_head1": labels_head1,
            "labels_head2": labels_head2,
            "target_imgs": target_imgs,
            "source_imgs": source_imgs,
            "distances": distances,
            "add_cur_images": add_cur_images,
        }

    def pad_sequence(self, seq, max_length, pad_value=0):
        """Pads a sequence to the desired length using tensor operations"""
        # Ensure seq is a tensor
        if isinstance(seq, torch.Tensor):
            seq = seq.clone().detach()
        else:
            # Convert seq to a tensor if it's not already
            seq = torch.tensor(seq)

        # Calculate the padding length and create the padded sequence
        padding_length = max_length - len(seq)

        if padding_length > 0:
            # Create the padded tensor by concatenating the original tensor with padding values
            padded_seq = torch.cat([seq, torch.full((padding_length,), pad_value)])
        else:
            padded_seq = seq[:max_length]  # Truncate if necessary

        return padded_seq


class MCTSDataset(Dataset):
    def __init__(self, dataset, random_flips=True):
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
        self.dataset = dataset
        self.transform = transforms.Compose(transformations)

    def __len__(self):
        return len(self.dataset)

    def open_image(self, img_path):
        image = Image.open(img_path)
        return self.transform(image)

    def __getitem__(self, idx):
        example_data = self.dataset[idx]
        target_img = self.open_image(example_data["target_img_path"])
        source_img = target_img
        if example_data["add_cur_image"]:
            source_img = self.open_image(example_data["node_image_path"])
        input_ids = example_data["input_ids"]
        attention = torch.ones(len(input_ids))

        return (
            example_data["input_ids"],
            example_data["main_head_labels"],
            example_data["secondary_head_labels"],
            example_data["target_value"],
            attention,
            target_img,
            source_img,
            example_data["add_cur_image"],
        )


class CustomDataCollatorForMCTSTraining:
    def __init__(self, tokenizer, loss_padding_value=-100):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer(tokenizer.special_tokens_map["pad_token"])["input_ids"][
            0
        ]  # yes I know it's stupid there was a problem with getting it directly
        self.loss_padding_value = loss_padding_value

    def __call__(self, batch):
        # Find the longest sequence length in the batch
        max_length = max(len(input_ids) for input_ids, _, _, _, _, _, _, _, in batch)

        # Initialize lists to store padded values
        input_ids = []
        attention_mask = []
        labels_head1 = []
        labels_head2 = []
        target_imgs = []
        source_imgs = []
        target_values = []
        add_cur_images = []

        # Iterate over each sample in the batch
        for (
            input_id,
            main_head_ids,
            secondary_head_ids,
            target_value,
            attention,
            target_img,
            source_img,
            add_cur_image,
        ) in batch:
            # Pad input_ids and attention_mask using tensor operations
            input_ids.append(self.pad_sequence(input_id, max_length, pad_value=self.pad_token_id))
            attention_mask.append(self.pad_sequence(attention, max_length, pad_value=0))

            # Pad labels (both heads) using tensor operations
            labels_head1.append(self.pad_sequence(main_head_ids, max_length, pad_value=self.loss_padding_value))
            labels_head2.append(self.pad_sequence(secondary_head_ids, max_length, pad_value=self.loss_padding_value))
            target_imgs.append(target_img)
            source_imgs.append(source_img)
            target_values.append(target_value)
            add_cur_images.append(add_cur_image)

        # Convert lists to tensors
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels_head1 = torch.stack(labels_head1)
        labels_head2 = torch.stack(labels_head2)
        target_imgs = torch.stack(target_imgs)
        source_imgs = torch.stack(source_imgs)
        target_values = torch.tensor(target_values)
        add_cur_images = torch.tensor(add_cur_images)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_head1": labels_head1,
            "labels_head2": labels_head2,
            "target_imgs": target_imgs,
            "source_imgs": source_imgs,
            "target_values": target_values,
            "add_cur_images": add_cur_images,
        }

    def pad_sequence(self, seq, max_length, pad_value=0):
        """Pads a sequence to the desired length using tensor operations"""
        # Ensure seq is a tensor
        if isinstance(seq, torch.Tensor):
            seq = seq.clone().detach()
        else:
            # Convert seq to a tensor if it's not already
            seq = torch.tensor(seq)

        # Calculate the padding length and create the padded sequence
        padding_length = max_length - len(seq)

        if padding_length > 0:
            # Create the padded tensor by concatenating the original tensor with padding values
            padded_seq = torch.cat([seq, torch.full((padding_length,), pad_value)])
        else:
            padded_seq = seq[:max_length]  # Truncate if necessary

        return padded_seq

    def _pad_2d_array(self, seq, target_length, pad_value=0):
        pad_amount = target_length - seq.size(0)
        return F.pad(seq, (0, 0, 0, pad_amount))
