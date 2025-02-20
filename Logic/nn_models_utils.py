from torchvision import transforms
from PIL import Image
import torch


def get_image_embedding(model, image_path, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
        transforms.Resize((224, 224)),  # Resize if necessary
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalization for grayscale image
    ])
    # Open the image and convert directly to grayscale
    img = Image.open(image_path).convert("L")  # 'L' mode for grayscale (1 channel)
    img = transform(img).unsqueeze(0).to(device)  # Convert and add batch dimension
    with torch.no_grad():
        embedding = model(img).cpu().numpy().flatten()  # Flatten to make it 1D
    return embedding


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
