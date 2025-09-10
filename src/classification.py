import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

letter_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'R',
    14: 'S',
    15: 'T',
    16: 'U',
    17: 'V',
    18: 'X',
    19: 'Y',
    20: 'Z'
}

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ObjectClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class ObjectDataset(Dataset):
    def __init__(self, objs, transform=None):
        self.objs = objs 
        self.transform = transform
    
    def __len__(self):
        return len(self.objs)
    
    def __getitem__(self, idx):
        img = self.objs[idx]
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        return img


def get_digit_model(digit_ckpt, num_classes=10):
    model = ObjectClassifier(num_classes).to(device)
    state_dict = torch.load(digit_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_letter_model(letter_ckpt, num_classes=21):
    model = ObjectClassifier(num_classes).to(device)
    state_dict = torch.load(letter_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_digit(image_arrs, model, batch_size):
    dataset = ObjectDataset(image_arrs, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, batch_preds = torch.max(outputs, 1)
            preds.extend(batch_preds.cpu().numpy().tolist())
    return preds


def predict_letter(image_arrs, model, batch_size):
    dataset = ObjectDataset(image_arrs, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, batch_preds = torch.max(outputs, 1)
            preds.extend(batch_preds.cpu().numpy().tolist())

    letters = [letter_dict[p] for p in preds]
    return letters