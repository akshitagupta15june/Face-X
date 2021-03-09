import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
from models import *

# change device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# preprocess the input image
preprocess = {
    "train": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# load datasets
data_dir = "Datasets"
datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), preprocess[x])
    for x in ["train", "val"]
}
dataloader = {
    x: torch.utils.data.DataLoader(
        datasets[x], batch_size=16, shuffle=True, num_workers=0
    )
    for x in ["train", "val"]
}
datasets_size = {x: len(datasets[x]) for x in ["train", "val"]}
class_names = datasets["train"].classes
import pdb; pdb.set_trace()
print(class_names)
print(len(class_names))

# initialize model
model = NASNetAMobile(len(class_names)).to(device)

# initialize model parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train(model, criterion, optimizer, scheduler, num_epochs=25):
    # helper function to train model
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        metrics = {
            "loss": {"train": 0.0, "val": 0.0},
            "acc": {"train": 0.0, "val": 0.0},
        }

        for phase in ["train", "val"]:
            running_loss = 0.0
            running_corrects = 0.0

            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(dataloader[phase], ncols=100):
                # iterate through the datasets
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = outputs.max(dim=1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            metrics["loss"][phase] = running_loss / datasets_size[phase]
            metrics["acc"][phase] = running_corrects.double() / datasets_size[phase]

        print(
            "Loss: {:.4f} Acc: {:.4f} Val Loss: {:.4f} Val Acc: {:.4f}".format(
                metrics["loss"]["train"],
                metrics["loss"]["val"],
                metrics["acc"]["train"],
                metrics["acc"]["val"],
            )
        )

        # update best model weights
        if (
            metrics["acc"]["val"]
            + metrics["acc"]["train"]
            - metrics["loss"]["val"]
            - metrics["loss"]["train"]
            > best_acc - best_loss
        ):
            best_acc = metrics["acc"]["val"]
            best_loss = metrics["loss"]["val"]
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best weights updated")

        print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# train model
model = train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)

# save model weights
torch.save(model.state_dict(), "saved_model.pt")
