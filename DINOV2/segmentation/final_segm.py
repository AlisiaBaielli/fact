import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import dinov2.eval.segmentation.utils.colormaps as colormaps
from dinov2.models.vision_transformer import vit_large
import time

DATA_DIR = "/home/azywot/data/ADEChallengeData2016/"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "images", "training")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "annotations", "training")
VAL_IMAGE_DIR = os.path.join(DATA_DIR, "images", "validation")
VAL_MASK_DIR = os.path.join(DATA_DIR, "annotations", "validation")

TEST_IMAGE_PATH = "/home/azywot/DINOv2/ADE_val_00001112.jpg"

DINO_PRETRAINED_PATH = "/home/azywot/DINOv2/dinov2_vitl14_pretrain.pth"
DINO_REG4_PRETRAINED_PATH = "/home/azywot/DINOv2/dinov2_vitl14_reg4_pretrain.pth"

BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 0.3
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 151
RESIZE_SIZE = (518, 518)

class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        image = TF.resize(image, RESIZE_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
        image = TF.to_tensor(image)

        mask = Image.open(mask_path)
        mask = TF.resize(mask, RESIZE_SIZE, interpolation=TF.InterpolationMode.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

train_dataset = ADE20KDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
val_dataset = ADE20KDataset(VAL_IMAGE_DIR, VAL_MASK_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class LinearHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=NUM_CLASSES, output_size=RESIZE_SIZE):
        super().__init__()
        self.decoder = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        self.output_size = output_size

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.decoder(x)
        return nn.functional.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

class SegmentationModel(nn.Module):
    def __init__(self, backbone_checkpoint, use_registers):
        super().__init__()
        self.backbone = vit_large(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=4 if use_registers else 0
        )

        state_dict = torch.load(backbone_checkpoint, map_location=DEVICE)
        self.backbone.load_state_dict(state_dict)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = LinearHead(in_dim=self.backbone.embed_dim, num_classes=NUM_CLASSES, output_size=RESIZE_SIZE)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(x, n=1)[0]
        return self.head(features)

def evaluate(model, loader):
    model.eval()
    num_classes = NUM_CLASSES
    total_inter = torch.zeros(num_classes, device=DEVICE)
    total_union = torch.zeros(num_classes, device=DEVICE)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)

            for cls in range(num_classes):
                mask_cls = (masks == cls)
                pred_cls = (preds == cls)
                inter = (mask_cls & pred_cls).sum().float()
                union = (mask_cls | pred_cls).sum().float()

                total_inter[cls] += inter
                total_union[cls] += union

    iou = total_inter / (total_union + 1e-6)
    mean_iou = iou.mean().item()
    return mean_iou

def train_model(model, train_loader, model_name, timestamp):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(model.module.head.parameters(), lr=LR, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, masks in loop:
            images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        miou = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Validation mIoU: {miou:.4f}")

    model_save_path = f"trained_model_{model_name.replace(' ', '_')}_{timestamp}.pth"
    torch.save(model.module.state_dict(), model_save_path)
    print(f"Saved trained model to {model_save_path}")

def predict_and_save(model, image_path, save_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = TF.resize(image, RESIZE_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
    image = TF.to_tensor(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    color_map = np.array(colormaps.ADE20K_COLORMAP, dtype=np.uint8)
    colored_pred = color_map[pred + 1]
    Image.fromarray(colored_pred).save(save_path)
    print(f"Segmentation saved at {save_path}")

def run_experiment(use_registers):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = "With Registers" if use_registers else "No Registers"
    checkpoint = DINO_REG4_PRETRAINED_PATH if use_registers else DINO_PRETRAINED_PATH

    print(f"\n==== Training model: {model_name} ====")
    model = SegmentationModel(checkpoint, use_registers=use_registers).to(DEVICE)
    model = nn.DataParallel(model)

    train_model(model, train_loader, model_name, timestamp)

    print(f"\nEvaluating {model_name} on validation set...")
    miou_val = evaluate(model, val_loader)
    print(f"{model_name} - Validation mIoU: {miou_val:.4f}")

    save_path = f"predicted_mask_{model_name.replace(' ', '_')}_{timestamp}.png"
    predict_and_save(model, TEST_IMAGE_PATH, save_path)
    print(f"Saved prediction for {model_name} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registers", action="store_true", help="Use model with register tokens")
    args = parser.parse_args()

    run_experiment(use_registers=args.registers)