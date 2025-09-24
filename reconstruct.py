import json
import re
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pickle
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import random_split

device = torch.device("cuda")
patch_size = 16
image_size = 224
mask_ratio = 0.3

class SimpleViTEncoder(nn.Module):
    def __init__(self, num_patches=196, patch_dim=256, embed_dim=128, depth=152, heads=4):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth) 
        
        self.reconstruct = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_dim)
        )

    def forward(self, x):
        x = self.proj(x) + self.pos_embedding
        x = self.transformer(x)
        return self.reconstruct(x)

model = SimpleViTEncoder().to(device)
model.load_state_dict(torch.load("/home/ml-lab/Autoencoder-CXR/vit_encoder_xray.pth", map_location=device))
model.eval()


def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 1, H, W]
    return img_tensor



def patchify(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    return patches.permute(0, 2, 1, 3, 4).flatten(2)  # [B, N, patch_dim]

def random_masking(patches, mask_ratio=0.3):
    B, N, D = patches.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=patches.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([B, N], dtype=torch.bool, device=patches.device)
    for i in range(B):
        mask[i, ids_keep[i]] = False
    patches_masked = patches.clone()
    patches_masked[mask] = 0
    return patches, patches_masked, mask


def show_masked_input(original, masked, mask, patch_size=16, image_size=224):
    """
    Visualizes the masked version of the input image before feeding to the model.
    """
    B, N, D = original.shape
    grid_size = int(N ** 0.5)
    patch_dim = int(D ** 0.5)

    # Restore masked patches into image grid
    patches = masked.view(B, N, 1, patch_dim, patch_dim)
    patches = patches.permute(0, 2, 1, 3, 4)  # [B, C, N, H, W]
    patches = patches.view(B, 1, grid_size, grid_size, patch_dim, patch_dim)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    img_masked = patches.view(B, 1, image_size, image_size).squeeze().cpu()

    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img_masked, cmap='gray')
    plt.title("Masked Input")
    plt.axis('off')
    plt.show()
    plt.savefig('plot_CXR_masked_1.png')



import matplotlib.pyplot as plt

def reconstruct_and_show(image_path):
    img = process_image(image_path)
    patches = patchify(img)
    original, masked, mask = random_masking(patches, mask_ratio)
    show_masked_input(original, masked, mask)

    with torch.no_grad():
        output = model(masked)

    reconstructed = original.clone()
    reconstructed[mask] = output[mask]

    B, N, D = reconstructed.shape
    patch_dim = int(D ** 0.5)
    grid_size = int(N ** 0.5)
    patches = reconstructed.view(B, N, 1, patch_dim, patch_dim)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.view(1, 1, grid_size, grid_size, patch_dim, patch_dim)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    img_recon = patches.view(1, 1, image_size, image_size).squeeze().cpu()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img.squeeze().cpu(), cmap='gray')
    axs[0].set_title("Original")
    axs[1].imshow(img_recon, cmap='gray')
    axs[1].set_title("Reconstructed")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('plot_CXR_1.png')


reconstruct_and_show("/home/ml-lab/Autoencoder-CXR/selected-images/s50045472.jpg")