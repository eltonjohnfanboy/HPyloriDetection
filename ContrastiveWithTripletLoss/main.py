import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import json
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from dataset import TripletDataset
import random
from triplet_loss import TripletLoss
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from model import ResNet_Triplet
from tqdm import tqdm

train_data_path = 'Data/AnnotatedPatches'
train_data = pd.read_csv('Data/window_metadata_sampled.csv')
train_set, test_set = train_test_split(train_data, test_size=0.1, random_state=42)

def get_train_dataset(IMAGE_SIZE=256):
    train_dataset = TripletDataset(train_data,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))]))
    test_dataset = TripletDataset(test_set,path=train_data_path,train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))]))    
    return train_dataset

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

IMAGE_SIZE = 28
BATCH_SIZE = 64
DEVICE = get_device()
LEARNING_RATE = 0.001
EPOCHS = 200

train_dataset = get_train_dataset(IMAGE_SIZE = IMAGE_SIZE)
train_dl = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

ResNet = ResNet_Triplet()
ResNet = ResNet.to(DEVICE)
optimizer = torch.optim.Adam(ResNet.parameters(),lr = LEARNING_RATE)
criterion = TripletLoss()
losses_train = []
losses_val =  []

for epoch in tqdm(range(EPOCHS), desc='Epochs'):
    ResNet.train()
    loss_acum = []
    for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_dl, desc='Training', leave=False)):
        if step == 4:
            test_anchor_img = anchor_img
            test_positive_img = positive_img
            test_negative_img = negative_img
            continue
        anchor_img = anchor_img.to(DEVICE)
        positive_img = positive_img.to(DEVICE)
        negative_img = negative_img.to(DEVICE)
        optimizer.zero_grad()
        anchor_out = ResNet(anchor_img)
        positive_out = ResNet(positive_img)
        negative_out = ResNet(negative_img)
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        loss_acum.append(loss.cpu().detach().numpy())
    losses_train.append(np.mean(loss_acum))
    ResNet.eval()
    test_anchor_img = test_anchor_img.to(DEVICE)
    test_positive_img = test_positive_img.to(DEVICE)
    test_negative_img = test_negative_img.to(DEVICE)
    anchor_out = ResNet(test_anchor_img)
    positive_out = ResNet(test_positive_img)
    negative_out = ResNet(test_negative_img)
    loss = criterion(anchor_out, positive_out, negative_out)
    losses_val.append(np.mean(loss.cpu().detach().numpy()))
    print('Epoch: {}/{} — Loss: {:.4f}\n'.format(epoch+1, EPOCHS, np.mean(loss_acum)))

embeddings = ResNet.Feature_Extractor(anchor_img)

embeddings = ResNet.Feature_Extractor(anchor_img)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings.detach().numpy())

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=anchor_label.detach().tolist(), cmap='viridis', marker='o')
plt.title('Visualització t-SNE dels Embedings 64-Dimensionals')
plt.colorbar()
plt.show()