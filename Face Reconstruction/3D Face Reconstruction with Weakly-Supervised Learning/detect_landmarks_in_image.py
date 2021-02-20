from preprocess.mtcnn import MTCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import os


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


batch_size = 1
workers = 0 if os.name == 'nt' else 8
dataset_dir = r'facebank'
cropped_dataset = r'dataset'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=(300, 300), margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

dataset = datasets.ImageFolder(
    dataset_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(dataset_dir, cropped_dataset))
    for p, _ in dataset.samples
]
loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=collate_pil
)

for i, (x, y) in enumerate(loader):
    x = mtcnn(x, save_path=y, save_landmarks=True)
