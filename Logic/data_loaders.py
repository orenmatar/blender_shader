import logging
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image


class SiameseDataset(Dataset):
    def __init__(self, data_pairs, images_folder, resize=True, with_names=False):
        """
        Args:
            data_pairs (list of tuples): Each tuple is (image1_path, image2_path, label).
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


# Function to split dataset and create dataloaders
def create_dataloaders(image_pairs, images_path, test_size=0.2, batch_size=32, num_workers=4, persistent_workers=True, **kwargs):
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


def evaluate_model_by_attribute(model, test_loader, criterion):
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

    auc_scores = {'running_loss': running_loss}
    true_scores = all_scores["similar_pairs"]
    true_labels = [1] * len(true_scores)
    for key in [key for key in all_scores if key != "similar_pairs"]:
        false_scores = all_scores[key]
        false_labels = [0] * len(false_scores)
        auc_scores[key] = roc_auc_score(false_labels + true_labels, false_scores + true_scores)

    logging.info(f"Eval took: {round(time.time() - t,2)}")
    model.train()
    return auc_scores


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive
