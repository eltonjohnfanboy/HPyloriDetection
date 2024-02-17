import torch
import pandas as pd
from dataset import DatasetPatients
from torchvision import transforms
from ae import AEanomaly
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import KFold

# Definim dataset
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
csv_train = 'train_metadata.csv'
data_dir = '/fhome/mapsiv/QuironHelico/CroppedPatches'
ds = DatasetPatients(data_dir, csv_train, transforms=data_transform)

# Autoencoder entrenat
state_dict = torch.load('model25E.pth')['model_state_dict']
model = AEanomaly()
model.load_state_dict(state_dict)
model.cuda()
model.eval()

def red_pixels(img_array):
    pil_image = Image.fromarray(np.uint8(img_array.transpose(1, 2, 0) * 255)).convert('RGB')
    hsv_img = pil_image.convert('HSV')
    pixels = list(hsv_img.getdata())
    red_like_range = (-20, 20)
    red_like_pixel_count = sum(1 for pixel in pixels if red_like_range[0] <= pixel[0] <= red_like_range[1])
    return red_like_pixel_count

# Fem cv amb 5 folds
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=17)

for fold, (train_index, val_index) in enumerate(kf.split(ds)):
    print(f"Fold {fold + 1}/{num_folds}")

    train_dataset = torch.utils.data.Subset(ds, train_index)
    val_dataset = torch.utils.data.Subset(ds, val_index)

    train_patient_label = []
    train_patient_label_prediction = []
    train_percentatge_positius = {'POSITIVE': [], 'NEGATIVE': []}
    train_label_original = {'NEGATIVA': [], 'ALTA': [], 'BAIXA': []}
    train_per_or = []

    for imgs, p_label in train_dataset:
        results = []
        threshold = 298.6
        batch_size = 64
        for i in tqdm(range(0, len(imgs), batch_size), desc="Training - Predicting for patient"):
            image = torch.stack(imgs[i:i+batch_size])
            single_reconstructed = model(image.cuda())
            for j in range(single_reconstructed.shape[0]):
                red_pixels_reconstruction = red_pixels(single_reconstructed[j, :, :, :].detach().cpu().numpy())
                red_pixels_original = red_pixels(image[j,:,:,:].numpy())
                if red_pixels_original - red_pixels_reconstruction >= threshold:
                    results.append(1)
                else:
                    results.append(-1)

        results = np.array(results)
        num_pos = len(results[results == 1])
        if len(results != 0):
            train_patient_label.append('POSITIVE' if p_label in ['ALTA', 'BAIXA'] else 'NEGATIVE')
            if num_pos/len(results) > 0.05:
                train_patient_label_prediction.append('POSITIVE')
            else:
                train_patient_label_prediction.append('NEGATIVE')
            train_percentatge_positius[train_patient_label_prediction[-1]].append(num_pos/len(results))
            train_label_original[p_label].append(num_pos/len(results))
            train_per_or.append(num_pos/len(results))

    val_patient_label = []
    val_patient_label_prediction = []
    val_percentatge_positius = {'POSITIVE': [], 'NEGATIVE': []}
    val_label_original = {'NEGATIVA': [], 'ALTA': [], 'BAIXA': []}
    val_per_or = []

    for imgs, p_label in val_dataset:
        results = []
        threshold = 298.6
        batch_size = 64
        for i in tqdm(range(0, len(imgs), batch_size), desc="Validation - Predicting for patient"):
            image = torch.stack(imgs[i:i+batch_size])
            single_reconstructed = model(image.cuda())
            for j in range(single_reconstructed.shape[0]):
                red_pixels_reconstruction = red_pixels(single_reconstructed[j, :, :, :].detach().cpu().numpy())
                red_pixels_original = red_pixels(image[j,:,:,:].numpy())
                if red_pixels_original - red_pixels_reconstruction >= threshold:
                    results.append(1)
                else:
                    results.append(-1)

        results = np.array(results)
        num_pos = len(results[results == 1])
        if len(results != 0):
            val_patient_label.append('POSITIVE' if p_label in ['ALTA', 'BAIXA'] else 'NEGATIVE')
            if num_pos/len(results) > 0.05:
                val_patient_label_prediction.append('POSITIVE')
            else:
                val_patient_label_prediction.append('NEGATIVE')
            val_percentatge_positius[val_patient_label_prediction[-1]].append(num_pos/len(results))
            val_label_original[p_label].append(num_pos/len(results))
            val_per_or.append(num_pos/len(results))