import helpfunctions

import torch
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

TRAIN_DATA_DIR = r"C:\Users\ludvi\Documents\Code\tjena\chest_xray\train"
VAL_DATA_DIR = r"C:\Users\ludvi\Documents\Code\tjena\chest_xray\val"
TEST_DATA_DIR = r"C:\Users\ludvi\Documents\Code\tjena\chest_xray\test"

BATCH_SIZE = 7

transform = v2.Compose([
        v2.Resize(size=(224,224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

# Load dataset
train_set = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=transform)
val_set = datasets.ImageFolder(root=VAL_DATA_DIR,transform=transform)
test_set = datasets.ImageFolder(root=TEST_DATA_DIR,transform=transform)

# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

transToPIL = v2.ToPILImage()
img = transToPIL(train_set[0][0])
# img.show()