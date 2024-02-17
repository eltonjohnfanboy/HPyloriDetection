import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm

class DatasetPatients(Dataset):
    def __init__(self, data_dir, csv_path, transforms = None):
        self.data_dir = data_dir
        self.csv_data = pd.read_csv(csv_path)
        self.transforms = transforms

        self.patients = {}
        self.labels = []
        self.index_map = []

        for idx,(_,row) in enumerate(self.csv_data.iterrows()):
            self.patients[row[0]] = []
            self.labels.append(row[1])
            self.index_map.append(row[0])
        
        for k in self.patients.keys():
            folder_path = os.path.join(self.data_dir, k+'_1')
            if os.path.isdir(folder_path):
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith('.png'):
                            img_path = os.path.join(folder_path, img_name)
                            self.patients[k].append(img_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        patient = self.patients[self.index_map[idx]]
        p_class = self.labels[idx]

        images = []
        count = 0
        for i in range(len(patient)):
            count += 1
            img_path = patient[i]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transforms:
                image = self.transforms(image)
            
            images.append(image)

        return images, p_class


class HpyloriDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_data = pd.read_csv(csv_file)
        self.transform = transform
        self.negatives = 0

        self.image_paths = []
        self.labels = []

        for _, row in self.csv_data.iterrows():
            folder_name = row[0]
            img_class = row[1]
            if self.negatives == 33:
              break
            if img_class == 'NEGATIVA':
                folder_path = os.path.join(self.data_dir, folder_name+'_1')
                if os.path.isdir(folder_path):
                    self.negatives += 1
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith('.png'):
                            img_path = os.path.join(folder_path, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append(img_class)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_class = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, img_class

class HpyloriDatasetAnnotated(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_data = pd.read_csv(csv_file)
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for idx, row in self.csv_data.iterrows():
            img_filename = row[0]
            img_class = row[1]
            if img_class != 0:
                folder_name, img_name = img_filename.split('.')
                folder_path = os.path.join(self.data_dir, folder_name)
                img_path = os.path.join(folder_path, img_name+'.png')
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(img_class)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_class = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
          if self.transform:
              image = self.transform(image)
        except:
          print(img_path)
          print(img_class)
          #print(image)
          print("----")
          return None, None
        return image, img_class
