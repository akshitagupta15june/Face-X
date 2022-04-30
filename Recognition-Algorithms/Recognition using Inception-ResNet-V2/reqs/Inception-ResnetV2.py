#!/usr/bin/env python
# coding: utf-8

# # Inception-ResNet-V2 : Face Recognition
# 
# #### Developed by Szegedy et. al.
# #### Contributed by : Suvaditya Mukherjee [@suvadityamuk](https://github.com/suvadityamuk)

# ### Import Calls

# Imports include
# - `torch` : An open source machine learning framework that accelerates the path from research prototyping to production deployment.
# - `pandas` : pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
# - `tqdm` : tqdm is a library in Python which is used for creating Progress Meters or Progress Bars.
# - `PIL` : Python Imaging Library is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
# - `prefetch_generator` : Simple package that makes your generator work in background thread.
# - `time` : This module provides various time-related functions.

# In[ ]:


import torch
from torch import nn
from torch.nn import functional as F
import os
import pandas
from torchvision.io import read_image   
from torch import optim
from tqdm.notebook import tqdm_notebook
from prefetch_generator import BackgroundGenerator
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


# ## Implementation of each layer present in the paper

# ### Definition of custom LambdaScale

# Uses `jit.ScriptModule` to help convert the Python-based logic to C++-backed code for compatibility with TorchScript based saving functions

# In[ ]:


class LambdaScale(jit.ScriptModule):
    def __init__(self):
        super().__init__()
        
        self.lambda_f = lambda x:x*0.1
    def forward(self, X):
        X = self.lambda_f(X)
        return X


# ### Definition of Stem

# In[ ]:


class InceptionResnetv2Stem(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sub0conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.sub0conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.sub0conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        
        self.sub1p1_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.sub1p2_conv1 = nn.Conv2d(64, 80, kernel_size=3, stride=2)
        
        self.sub2p1_conv1 = nn.Conv2d(64, 80, kernel_size=1, padding='same')
        self.sub2p1_conv2 = nn.Conv2d(80, 192, kernel_size=3)
        
        self.sub3p2_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.branch0 = nn.Conv2d(192, 96, kernel_size=1)
        
        self.branch1a = nn.Conv2d(192, 48, kernel_size=1)
        self.branch1b = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        
        self.branch2a = nn.Conv2d(192, 64, kernel_size=1)
        self.branch2b = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch2c = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        
        self.branch3a = nn.AvgPool2d(3, padding=1, count_include_pad=False)
        self.branch3b = nn.Conv2d(192, 64, kernel_size=1, stride=1)
        
        self.batchNorm = nn.BatchNorm2d(320)
    
    def forward(self, X):
        
        X = F.relu(self.sub0conv1(X)) 
        X = F.relu(self.sub0conv2(X)) 
        X = F.relu(self.sub0conv3(X)) 
        
        X = self.sub1p1_mpool1(X)
        X = F.relu(self.sub2p1_conv1(X))
        X = F.relu(self.sub2p1_conv2(X))
        
        X = self.sub3p2_mpool1(X)
        
        X0 = self.branch0(X)
        
        X1 = self.branch1a(X)
        X1 = self.branch1b(X1)
        
        X2 = self.branch2a(X)
        X2 = self.branch2b(X2)
        X2 = self.branch2c(X2)
        
        X3 = self.branch3a(X)
        X3 = self.branch3b(X)
        
        X = torch.cat((X0, X1, X2, X3), 1)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        return X


# ### Definition of ResNet Block A

# In[ ]:


class InceptionResnetv2A(nn.Module):
    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale
        
        self.p1_conv1 = nn.Conv2d(320, 32, kernel_size=1, padding='same')
        
        self.p2_conv1 = nn.Conv2d(320, 32, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        
        self.p3_conv1 = nn.Conv2d(320, 32, kernel_size=1, padding='same')
        self.p3_conv2 = nn.Conv2d(32, 48, kernel_size=3, padding='same')
        self.p3_conv3 = nn.Conv2d(48, 64, kernel_size=3, padding='same')
        
        self.p_conv1 = nn.Conv2d(128, 320, kernel_size=1, padding='same')
        
        self.batchNorm = nn.BatchNorm2d(320, affine=True)
        
        if self.scale:
            self.scaleLayer = LambdaScale()
        
    def forward(self, X):
        
        # X is relu-activated
        old = X
        
        X1 = F.relu(self.p1_conv1(X))
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        
        X3 = F.relu(self.p3_conv1(X))
        X3 = F.relu(self.p3_conv2(X3))
        X3 = F.relu(self.p3_conv3(X3))
        
        X = torch.cat((X1, X2, X3), dim=1)
        
        X = self.p_conv1(X)
        if self.scale:
            X = self.scaleLayer(X)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X


# ### Definition of ResNet Block B

# In[ ]:


class InceptionResnetv2B(nn.Module):

    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale
        self.p1_conv1 = nn.Conv2d(1088, 192, kernel_size=1, stride=1, padding='same')
        
        self.p2_conv1 = nn.Conv2d(1088, 128, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(128, 160, kernel_size=(1,7), padding='same')
        self.p2_conv3 = nn.Conv2d(160, 192, kernel_size=(7,1), padding='same')
        
        self.p3_conv = nn.Conv2d(384, 1088, kernel_size=1, padding='same')
        
        self.batchNorm = nn.BatchNorm2d(1088, affine=True)
        if self.scale:
            self.scaleLayer = LambdaScale()
            
    def forward(self, X):
        old = X
        X1 = F.relu(self.p1_conv1(X))
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        X2 = F.relu(self.p2_conv3(X2))
        
        X = torch.cat((X1, X2), dim=1)
        
        X = F.relu(self.p3_conv(X))
        if self.scale:
            X = self.scaleLayer(X)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X


# ### Definition of ResNet Block C

# In[ ]:


class InceptionResnetv2C(nn.Module):
    def __init__(self, scale=True, noRelu=False):
        super().__init__()
        self.scale = scale
        
        self.noRelu = noRelu
        self.p1_conv1 = nn.Conv2d(2080, 192, kernel_size=1, padding='same')
        
        self.p2_conv1 = nn.Conv2d(2080, 192, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(192, 224, kernel_size=(1,3), padding='same')
        self.p2_conv3 = nn.Conv2d(224, 256, kernel_size=(3,1), padding='same')
        
        self.p3_conv = nn.Conv2d(448, 2080, kernel_size=1, padding='same')
        
        self.batchNorm = nn.BatchNorm2d(2080, affine=True)
        if self.scale:
            self.scaleLayer = LambdaScale()
    def forward(self, X):
        old = X
        X1 = F.relu(self.p1_conv1(X))
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        X2 = F.relu(self.p2_conv3(X2))
        
        X = torch.cat((X1, X2), dim=1)
        
        X = F.relu(self.p3_conv(X))
        if self.scale:
            X = self.scaleLayer(X)
        
        X = self.batchNorm(X)
        if not self.noRelu:
            X = F.relu(X)
        
        return X


# ### Definition of ResNet Block - Reduction A

# In[ ]:


class InceptionResnetv2ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.p1_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.p2_conv1 = nn.Conv2d(320, 384, kernel_size=3, stride=2)
        
        self.p3_conv1 = nn.Conv2d(320, 256, kernel_size=1, padding='same')
        self.p3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.p3_conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=2)
        
        self.batchNorm = nn.BatchNorm2d(1088, affine=True)
        
    def forward(self, X):
        
        X1 = self.p1_mpool1(X)
        
        X2 = F.relu(self.p2_conv1(X))
        
        X3 = F.relu(self.p3_conv1(X))
        X3 = F.relu(self.p3_conv2(X3))
        X3 = F.relu(self.p3_conv3(X3))
        
        X = torch.cat((X1, X2, X3), dim=1)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X


# ### Definition of ResNet Block - Reduction B

# In[ ]:


class InceptionResnetv2ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.p1_mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.p2_conv1 = nn.Conv2d(1088, 256, kernel_size=1, padding='same')
        self.p2_conv2 = nn.Conv2d(256, 384, kernel_size=3, stride=2)
        
        self.p3_conv1 = nn.Conv2d(1088, 256, kernel_size=1, padding='same')
        self.p3_conv2 = nn.Conv2d(256, 288, kernel_size=3, stride=2)
        
        self.p4_conv1 = nn.Conv2d(1088, 256, kernel_size=1, padding='same')
        self.p4_conv2 = nn.Conv2d(256, 288, kernel_size=3, padding=1)
        self.p4_conv3 = nn.Conv2d(288, 320, kernel_size=3, stride=2)
        
        self.batchNorm = nn.BatchNorm2d(2080, affine=True)
        
    def forward(self, X):
        
        X1 = self.p1_mpool1(X)
        
        X2 = F.relu(self.p2_conv1(X))
        X2 = F.relu(self.p2_conv2(X2))
        
        X3 = F.relu(self.p3_conv1(X))
        X3 = F.relu(self.p3_conv2(X3))
        
        X4 = F.relu(self.p4_conv1(X))
        X4 = F.relu(self.p4_conv2(X4))
        X4 = F.relu(self.p4_conv3(X4))
        
        X = torch.cat((X1, X2, X3, X4), dim=1)
        
        X = self.batchNorm(X)
        X = F.relu(X)
        
        return X


# ### Definition of final Model

# In[ ]:


class InceptionResnetV2(nn.Module):
    def __init__(self, scale=True, feature_list_size=1001):
        super().__init__()
        
        self.scale = scale
        self.stem = InceptionResnetv2Stem()
        self.a = InceptionResnetv2A(scale=True)
        self.b = InceptionResnetv2B(scale=True)
        self.c = InceptionResnetv2C(scale=True)
        self.noreluc = InceptionResnetv2C(scale=True, noRelu=True)
        self.red_a = InceptionResnetv2ReductionA()
        self.red_b = InceptionResnetv2ReductionB()
        
        self.avgpool = nn.AvgPool2d(8)
        
        self.conv2d = nn.Conv2d(2080, 1536, kernel_size=1,)
        
        self.dropout = nn.Dropout(0.8)
        self.flatten = nn.Flatten()
        
        self.linear = nn.Linear(in_features=1536, out_features=feature_list_size)
        
    
    def forward(self, X):
        X = self.stem(X)
        
        for i in range(10):
            X = self.a(X)
        
        X = self.red_a(X)
        
        for i in range(20):
            X = self.b(X)
        
        X = self.red_b(X)
        
        for i in range(9):
            X = self.c(X)
            
        X = self.noreluc(X)
        
        X = self.conv2d(X)
        
        X = self.dropout(X)
        
        X = self.avgpool(X)
        
        X = X.view(X.size(0), -1)
        
        X = self.linear(X)
        
        return X
        


# ### Test run of a random Tensor through the model

# This model takes a 299x299x3 image or Tensor as input. We now test the construction of the model by passing a randomly-generated tensor of the required dimensions

# In[ ]:


X = torch.randn(1, 3, 299, 299)
model = InceptionResnetV2(feature_list_size=7)
model.forward(X)


# ### Getting details of GPU present on machine and defining helpers to load previous models

# Steps include
# - Running `!nvidia-smi` to get the details of the NVIDIA GPUs present on the system. (only for GPU-backed systems)  
# - Using `torch.cuda.device_count()` to get the number of devices `torch` could use to perform operations. In case of a CPU + GPU system, we will see 2 devices. We use this information to set a variable `device` having information on the active device to use for all training operations later ahead  
# - Helpers to read in and use pre-trained versions of the model  

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


torch.cuda.device_count()


# In[ ]:


def try_gpu_else_cpu():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
device = try_gpu_else_cpu()


# In[ ]:


def load_model_from_checkpoint(path, size):
    res = torch.load(path)
    model = InceptionResnetV2(feature_list_size=size)
    model.load_state_dict(res['model.state_dict'])
    optimizer = optim.Adam(net.parameters(), weight_decay=0.009, amsgrad=True)
    optimizer.load_state_dict(res['optimizer.state_dict'])
    epoch = res['epoch']
    return model, optimizer, epoch


# ### Original-paper specified the following parameters

# In[ ]:


optimizer = optim.RMSprop(model.parameters(), weight_decay=0.9, eps=1.0, lr=0.045)
loss_fn = nn.CrossEntropyLoss()


# ### Dataset preprocessing and Model training
# 
# #### If you wish to download the dataset from Kaggle, you must generate an API Token first and place it in ~/.kaggle (Linux) or at C:/Users/(username) (Windows)
# #### The CelebA dataset also comes as a pre-processed dataset from PyTorch's `torchvision` Computer Vision library. More details can be found [here](https://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html)
# 
# Steps to be followed here:
# - Get the dataset onto local system
# - Prepare the dataset's labels by getting the .csv files and processing them
# - Prepare a custom extension of the `torch.utils.data.Dataset` for our current dataset
# - Prepare the image transformations necessary to work on the images using `torchvision.transforms`

# In[ ]:


get_ipython().system('pip install --no-deps --upgrade --force-reinstall kaggle')
get_ipython().system('mkdir ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/kaggle.json')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
get_ipython().system('kaggle datasets download -d jessicali9530/celeba-dataset')
get_ipython().system('unzip celeba-dataset')


# In[ ]:


get_ipython().system('mv img_align_celeba/img_align_celeba ..')


# In[ ]:


import pandas as pd
df_at = pd.read_csv("list_attr_celeba.csv")
df_div = pd.read_csv("list_eval_partition.csv")


# In[ ]:


df_image_train = pd.DataFrame()
df_image_test = pd.DataFrame()
df_image_valid = pd.DataFrame()
df_image_train['image_id'] = df_div[df_div['partition'] == 0]['image_id']
df_image_test['image_id'] = df_div[df_div['partition'] == 1]['image_id']
df_image_valid['image_id'] = df_div[df_div['partition'] == 2]['image_id']
df_image_train.reset_index(drop=True, inplace=True)
df_image_test.reset_index(drop=True, inplace=True)
df_image_valid.reset_index(drop=True, inplace=True)


# In[ ]:


get_gender = lambda x: df_at.iloc[x]['Male']


# In[ ]:


train_labels = list()
test_labels = list()
valid_labels = list()
for i in range(df_image_train.shape[0]):
    train_labels.append(get_gender(i))
df_image_train['gender'] = train_labels
for i in range(df_image_test.shape[0]):
    test_labels.append(get_gender(i))
df_image_test['gender'] = test_labels
for i in range(df_image_valid.shape[0]):
    valid_labels.append(get_gender(i))
df_image_valid['gender'] = valid_labels


# In[ ]:


df_image_train.to_csv('train_labels.csv', index=None)
df_image_test.to_csv('test_labels.csv', index=None)
df_image_valid.to_csv('valid_labels.csv', index=None)


# In[ ]:


print(f'Device: {device}')


# In[ ]:


import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class FaceRecognitionDataset(Dataset):
    def __init__(self, img_dir, img_labels, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(img_labels)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_classes = ['Female', 'Male']
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = read_image(img_path).type(torch.float32)
        label = 1 if self.img_labels.iloc[idx, 1] == 1 else 0
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


# In[ ]:


import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

IMG_DIR_PATH = "img_align_celeba" # Set this to your own folder which stores the dataset images
TR_IMG_LBL_PATH = "train_labels.csv"
TE_IMG_LBL_PATH = "test_labels.csv"
VL_IMG_LBL_PATH = "valid_labels.csv"

train_transforms = transforms.Compose([transforms.Resize(size=(299,299), interpolation=transforms.InterpolationMode.NEAREST), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = FaceRecognitionDataset(img_dir=IMG_DIR_PATH, img_labels=TR_IMG_LBL_PATH, transform=train_transforms)

data_train = DataLoader(train_dataset, shuffle=True, batch_size=9)


# ### Getting CUDA Memory summary and usage diagnostics

# In[ ]:


print(torch.cuda.memory_summary(device=device, abbreviated=False))


# ### Cleaning all previous cache before using GPU

# In[ ]:


torch.cuda.empty_cache()


# ### Setting all seeds and options required to maintain reproducibility

# In[ ]:


torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)


# ### Training function

# In[ ]:


def train_net(train_loader, epochs=2):
    
    CURRENT_DIRECTORY = os.getcwd()
    EPOCH_DIRECTORY = os.path.join(CURRENT_DIRECTORY, 'resnet-v2-epochs')
    if not os.path.exists(EPOCH_DIRECTORY):
        os.mkdir(EPOCH_DIRECTORY)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = InceptionResnetV2(feature_list_size=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.009, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    
    running_loss = 0.00
    count = 0
    
    writer = SummaryWriter()
    
    for i in range(epochs):
        
        pbar = tqdm_notebook(enumerate(BackgroundGenerator(train_loader), 0),
                    total=len(train_loader))
        start_time = time.time()
        
        CHECKPOINT_PATH = os.path.join(EPOCH_DIRECTORY, f'model_ckpt_epoch{i+1}.pkl')
        TORCHSCRIPT_PATH = os.path.join(EPOCH_DIRECTORY, f'script_resnetv2_model_{i+1}.pkl')
        
        for j, data in pbar:
            images, labels = data
            
            inp, targs = images.to(device), labels.to(device)
                
            prepare_time = start_time-time.time()

            optimizer.zero_grad()

            output = net(inp)
            loss = loss_fn(output, targs)
            loss.backward()
            optimizer.step()
            count+=1
            
            process_time = start_time-time.time()-prepare_time
            pbar.set_description(f'Efficiency = {process_time/(process_time+prepare_time):.4f}\tEpochs: {i+1}/{epochs}\tLoss: {loss:.4f}')
            running_loss += loss.item()
            
            writer.add_scalar('Compute Time efficiency (per mini-batch)', process_time/(process_time+prepare_time),
                             j)
            writer.add_scalar('Training Loss', loss, j)
            
            
            
        scheduler.step(loss)
        torch.save({
            "model.state_dict" : net.state_dict(),
            "optimizer.state_dict" : optimizer.state_dict(),
            "epoch":i
        }, CHECKPOINT_PATH)
    
    
    writer.flush()
    writer.close()
    
    img, lbl = next(iter(train_loader))
    img = img.to(device)
    writer.add_graph(net, img)
    
    import gc
    gc.collect()

    return net, optimizer


# The below function begins the training process. Change the `epochs` parameter on line 4 to train for higher epochs. A good number to train for would be 10. The model after each epoch, will be saved as a .pkl file for future use, with the final model saved as a TorchScript model file for high-speed inferencing

# In[ ]:


CURRENT_DIRECTORY = os.getcwd()
EPOCH_DIRECTORY = os.path.join(CURRENT_DIRECTORY, 'resnet-v2-epochs')
TORCHSCRIPT_PATH = os.path.join(EPOCH_DIRECTORY, f'script_resnetv2_model_{i+1}.pkl')
net, opt = train_net(data_train, epochs=1)
net_scripted = torch.jit.script(net)
net_scripted.save(TORCHSCRIPT_PATH)


# ### Defining function to load TorchScript-based model. Use as per needed, if required.

# In[ ]:


def loadTorchScript(SCRIPT_PATH):
    net = torch.jit.load(SCRIPT_PATH)
    net.eval()
    return net


# ### Using Tensorboard. Navigate to http://localhost:6006/ while cell is executing

# TensorBoard logs the per-minibatch loss, as well as a full visualization of the model in an interactive manner

# In[ ]:


# View Tensorboard
get_ipython().system('pip install tensorboard')
get_ipython().system('tensorboard --logdir=runs')


# ### Defining functions to generate predictions from model

# In[ ]:


def predict_class(img, transform_func):
    classes = ['Female', 'Male']
    var = torch.autograd.Variable(img).cuda()
    
    # Use latest model epoch by changing path
    # model, opt, ep = load_model_from_checkpoint("model_ckpt_epochn.pkl")
    model = net
    res = model(var)
    res = res.cpu()
    clsf = res.data.numpy()
    pred = list()
    print(res)
    print(clsf)
    for i in clsf:
        print(i.argmax())
        pred.append(classes[i.argmax()])
    return pred


# ### Testing phase

# In[ ]:


test_transforms = transforms.Compose([transforms.Resize(size=(299,299), interpolation=transforms.InterpolationMode.NEAREST), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = FaceRecognitionDataset(img_dir=IMG_DIR_PATH, img_labels=TE_IMG_LBL_PATH, transform=test_transforms)

data_test = DataLoader(test_dataset, shuffle=True, batch_size=2)

images, labels = next(iter(data_test))

for i in range(len(images)):

    img = images.cpu().numpy()[i]
    img = img.astype(int)
    img = np.transpose(img, (1,2,0))
    plt.imshow(img)
    plt.show()  

    res = predict_class(images, test_transforms)
    print(labels)
    print(res)


# ### How to run the notebook to generate inferences
# 
# - To see the model in action in general, run all cells and see the inferences from the last cell.

# ### Closing Ideas
# 
# The training of this model requires over 10 epochs to produce viable results.
# Per-epoch training on a NVIDIA GTX 1650Ti 4GB, Intel i7-10750H, 16GB RAM computer required approximately 2 hours with a batch-size of 9.
# On using Colab Pro Plus with NVIDIA Tesla GPUs, we got a training time of approximately 45 minutes with the training samples present.
# For better use, one can fork, make changes and push pre-trained checkpoint files of the model for better and easier direct use.
