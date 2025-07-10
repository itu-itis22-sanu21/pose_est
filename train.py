'''import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from models.with_mobilenet import LightOpenPose
from penn_action import PennActionDataset, PennTransform

def generate_heatmaps(keypoints, height, width, sigma=2):
    device = keypoints.device  # ensure all computations happen on the same device

    num_keypoints = keypoints.shape[1]
    heatmaps = torch.zeros((keypoints.shape[0], num_keypoints, height, width), device=device, dtype=torch.float32)
    visibility_mask = torch.zeros_like(heatmaps)

    # Precompute meshgrid once and send to device
    xx, yy = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing='xy'
    )

    for i in range(keypoints.shape[0]):  # batch
        for j in range(num_keypoints):
            x, y, v = keypoints[i, j]
            if v > 0:
                center_x = x * width
                center_y = y * height
                heatmap = torch.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
                heatmaps[i, j] = heatmap
                visibility_mask[i, j] = 1.0

    return heatmaps, visibility_mask


def l2_loss(input, target, mask, batch_size):
    
    # loss = (input - target) * mask
    # loss = (loss * loss) / 2 / batch_size
    
    loss = ((input - target)**2 * mask).sum() / batch_size

    return loss.sum()

def compute_pck(pred_heatmaps, true_heatmaps, threshold=0.2):
    B, J, H, W = pred_heatmaps.shape
    pred_coords = torch.argmax(pred_heatmaps.view(B, J, -1), dim=2)
    true_coords = torch.argmax(true_heatmaps.view(B, J, -1), dim=2)

    pred_y = pred_coords // W
    pred_x = pred_coords % W
    true_y = true_coords // W
    true_x = true_coords % W

    dist = torch.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2).float()
    norm = torch.tensor([H, W]).float().mean()
    pck = (dist < threshold * norm).float().mean()
    return pck.item()

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_pck = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            gt_heatmaps, mask = generate_heatmaps(keypoints, height=128, width=128)
            gt_heatmaps = gt_heatmaps.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = sum([l2_loss(out, gt_heatmaps, mask, images.size(0)) for out in outputs])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pck += compute_pck(outputs[-1], gt_heatmaps)

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        avg_pck = epoch_pck / len(train_loader)
        print(f"\n✅ Epoch [{epoch+1}] finished in {time.time() - start_time:.2f}s")
        print(f"   🔥 Avg Train Loss: {avg_loss:.6f}")
        print(f"   🎯 Train PCK Accuracy: {avg_pck:.4f}")
        validate_model(model, val_loader, device)

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_pck = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            gt_heatmaps, mask = generate_heatmaps(keypoints, height=128, width=128)
            gt_heatmaps = gt_heatmaps.to(device)
            mask = mask.to(device)

            outputs = model(images)
            loss = sum([l2_loss(out, gt_heatmaps, mask, images.size(0)) for out in outputs])
            pck = compute_pck(outputs[-1], gt_heatmaps)

            val_loss += loss.item()
            val_pck += pck

    avg_val_loss = val_loss / len(val_loader)
    avg_val_pck = val_pck / len(val_loader)
    print(f"   📉 Validation Loss: {avg_val_loss:.6f}")
    print(f"   🎯 Validation PCK Accuracy: {avg_val_pck:.4f}\n")
    model.train()

def test_model(model, test_loader, device):
    model.eval()
    print("Testing on test set...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            outputs = model(images)
            print(f"Test batch {i+1}: Output shape {outputs[-1].shape}")
            if i >= 1:
                break

if __name__ == '__main__':
    annotation_dir = 'Penn_Action/labels'
    image_root = 'Penn_Action/frames'

    dataset = PennActionDataset(annotation_dir, image_root, transform=PennTransform())
    print(f"Loaded {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightOpenPose(num_refinement_stages=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5)

    print("Saving model...")
    torch.save(model.state_dict(), "penn_pose_model.pth")

    test_model(model, test_loader, device)
'''

# train.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

from models.with_mobilenet import PoseEstimationWithMobileNet
from penn_action import PennActionDataset, PennTransform

def generate_heatmaps(keypoints, height, width, sigma=2):
    num_keypoints = keypoints.shape[1]
    heatmaps = torch.zeros((keypoints.shape[0], num_keypoints, height, width), dtype=torch.float32, device=keypoints.device)
    visibility_mask = torch.zeros_like(heatmaps)

    for i in range(keypoints.shape[0]):  # batch
        for j in range(num_keypoints):
            x, y, v = keypoints[i, j]
            if v > 0:
                xx, yy = torch.meshgrid(torch.arange(width, device=keypoints.device), torch.arange(height, device=keypoints.device), indexing='xy')
                heatmap = torch.exp(-((xx - (x * width))**2 + (yy - (y * height))**2) / (2 * sigma**2))
                heatmaps[i, j] = heatmap
                visibility_mask[i, j] = 1.0
    return heatmaps, visibility_mask

def compute_pck(pred_heatmaps, true_heatmaps, threshold=0.2):
    B, J, H, W = pred_heatmaps.shape
    pred_coords = torch.argmax(pred_heatmaps.view(B, J, -1), dim=2)
    true_coords = torch.argmax(true_heatmaps.view(B, J, -1), dim=2)

    pred_y = pred_coords // W
    pred_x = pred_coords % W
    true_y = true_coords // W
    true_x = true_coords % W

    dist = torch.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2).float()
    norm = torch.tensor([H, W], dtype=torch.float32, device=dist.device).mean()
    pck = (dist < threshold * norm).float().mean()
    return pck.item()

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_pck = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            gt_heatmaps, visibility_mask = generate_heatmaps(keypoints, height=32, width=32)

            optimizer.zero_grad()
            outputs = model(images)
            pred_heatmaps = outputs[-1]

            # Use visibility-masked MSE loss
            loss = ((pred_heatmaps - gt_heatmaps)**2 * visibility_mask).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pck += compute_pck(pred_heatmaps, gt_heatmaps)

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        avg_pck = epoch_pck / len(train_loader)
        print(f"\n✅ Epoch [{epoch+1}] finished in {time.time() - start_time:.2f}s")
        print(f"   🔥 Avg Train Loss: {avg_loss:.6f}")
        print(f"   🎯 Train PCK Accuracy: {avg_pck:.4f}")
        validate_model(model, val_loader, device)

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_pck = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            gt_heatmaps, visibility_mask = generate_heatmaps(keypoints, height=128, width=128)

            outputs = model(images)
            pred_heatmaps = outputs[-1]
            loss = ((pred_heatmaps - gt_heatmaps)**2 * visibility_mask).mean()
            pck = compute_pck(pred_heatmaps, gt_heatmaps)

            val_loss += loss.item()
            val_pck += pck

    avg_val_loss = val_loss / len(val_loader)
    avg_val_pck = val_pck / len(val_loader)
    print(f"   📉 Validation Loss: {avg_val_loss:.6f}")
    print(f"   🎯 Validation PCK Accuracy: {avg_val_pck:.4f}\n")
    model.train()

def test_model(model, test_loader, device):
    model.eval()
    print("Testing on test set...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            outputs = model(images)
            print(f"Test batch {i+1}: Output shape {outputs[-1].shape}")
            if i >= 1:
                break

if __name__ == '__main__':
    annotation_dir = 'Penn_Action/labels'
    image_root = 'Penn_Action/frames'

    dataset = PennActionDataset(annotation_dir, image_root, transform=PennTransform())
    print(f"Loaded {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseEstimationWithMobileNet(num_refinement_stages=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, optimizer, device, num_epochs=5)

    print("Saving model...")
    torch.save(model.state_dict(), "penn_pose_model.pth")

    test_model(model, test_loader, device)
