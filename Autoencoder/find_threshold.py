import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from dataset import HpyloriDatasetAnnotated
from torchvision import transforms
from ae import AEanomaly
from statistics import mean

# Load autoencoder
state_dict = torch.load('fold_3_model.pth')['model_state_dict']
model = AEanomaly().cuda()
model.load_state_dict(state_dict)
model.eval()

data_dir = '/fhome/mapsiv/QuironHelico/AnnotatedPatches'
csv_file = '/fhome/mapsiv/QuironHelico/AnnotatedPatches/window_metadata.csv'

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Definim dataset
dataset = HpyloriDatasetAnnotated(data_dir, csv_file, transform=data_transform)

print(len(dataset))

# Ens guardem les dades i les labels
input_data = []
true_labels = []

for i in range(len(dataset)):
    image, img_class = dataset[i]
    if image is not None:
        input_data.append(image.detach().cpu().numpy())
        true_labels.append(img_class)

print("ye")
print(len(input_data))
print(len(true_labels))
print("DONE___")
input_data = np.array(input_data)
true_labels = np.array(true_labels)

# Predim amb l'autoencoder
mse_losses = np.array([])
batch_size = 64
for i in range(0, len(input_data), batch_size):
    # Processem un sample alhora
    single_input = torch.from_numpy(input_data[i:i+batch_size]).float().cuda()
    single_reconstructed = model(single_input).detach().cpu().numpy()

    # Calculem el MSE del sample
    single_mse = np.mean((input_data[i:i+batch_size] - single_reconstructed) ** 2, axis=(1, 2, 3))
    mse_losses = np.append(mse_losses, single_mse)

#input_tensor = torch.from_numpy(input_data).float().cuda()
#reconstructed_data = model(input_tensor).detach().numpy() #això no se si anirà xd
#mse_losses = np.mean((input_data - reconstructed_data) ** 2, axis=(1, 2, 3))

# Mean losses
print("MITJANA LOSSES")
print(mean(mse_losses))
print("Mitjana de losses pels que tenen (-1) és a dir, NO tenen la vaina")
print(mean(mse_losses[true_labels == -1]))
print("Mitjana de losses pels que tenen (1) és a dir, SI tenen la vaina")
print(mean(mse_losses[true_labels == 1]))

# Fem ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, mse_losses)

# Gmean per trobar el millor threshold
g_mean = np.sqrt(tpr * (1 - fpr))
millor_thr = thresholds[np.argmax(g_mean)]


# Plot de la roc curve
roc_auc = auc(fpr, tpr)
"""
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall (TPR)')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()
"""
