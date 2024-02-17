from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import HpyloriDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from ae import AEanomaly
from train import train_epoch
from test import test_epoch
from statistics import mean
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# Carreguem el metadata i fem els seus splits si no existeixen
if os.path.exists('train_metadata.csv') and os.path.exists('test_metadata.csv'):
    print("Splits ja existeixen.")
else:
    # Carreguem el metadata
    metadata = pd.read_csv('/fhome/mapsiv/QuironHelico/CroppedPatches/metadata.csv')

    # Split en 80/20
    train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=17)

    # Guardem els csv
    train_data.to_csv('train_metadata.csv', index=False)
    test_data.to_csv('test_metadata.csv', index=False)


# Defnim el dataset i les transformacions que li volem fer
data_dir = '/fhome/mapsiv/QuironHelico/CroppedPatches'
csv_file = 'train_metadata.csv'

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = HpyloriDataset(data_dir, csv_file, transform=data_transform)
print(len(dataset))
print(dataset.negatives)

# Defnim els hyperparams, loss function i el k-fold
EPOCHS = 25
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
stratified_kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 17)
history = []
#print(device)
#model = AEanomaly().to(device)
#print(next(model.parameters()).is_cuda)

# Iterem pels diferent folds (això no caldria, però hem fet un codi base per després poder-ho aprofitar)
#for fold, (train_idx,val_idx) in enumerate(stratified_kfold.split(dataset.image_paths, dataset.labels)):
    
#    print('Fold {}'.format(fold + 1))

# Defnim els dataloader
#train_sampler = SubsetRandomSampler(train_idx)
#test_sampler = SubsetRandomSampler(val_idx)
train_loader = DataLoader(dataset, batch_size=64)
test_loader = DataLoader(dataset, batch_size=64)

# Definim model + optimizer
model = AEanomaly()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 0.005)
model_perf = {'train_loss': [], 'test_loss': []}

# Entrenem model per cada fold
for e in range(EPOCHS):
    train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
    test_loss = test_epoch(model, device, test_loader, criterion)

    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)

    print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f}".format(e + 1,
                                                                            EPOCHS,
                                                                            train_loss,
                                                                            test_loss))
    model_perf['train_loss'].append(train_loss)
    model_perf['test_loss'].append(test_loss)

# Guardem el model per data fold
torch.save({
'epoch': EPOCHS,
'model_state_dict': model.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
'train_loss': model_perf['train_loss'], 
'test_loss': model_perf['test_loss'] 
}, 'model25E.pth')

#history.append((fold+1, fold_perf))

# Print de les losses pels diferents folds
#for e,i in enumerate(history):
#    print(f"Fold {e+1}")
#    print(f"Avg. train loss: {mean(i[1]['train_loss'])} - Avg. test loss: {mean(i[1]['test_loss'])}")
#    print("----")


