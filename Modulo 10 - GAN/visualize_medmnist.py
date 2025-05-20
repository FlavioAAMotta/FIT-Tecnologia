import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO, Evaluator
import torch
from torchvision import transforms
import numpy as np

# Carregando o dataset PathMNIST
data_flag = 'pathmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

print(f'Task: {task}')
print(f'Number of Classes: {n_classes}')
print(f'Labels: {info["label"]}')

# Carregando o dataset
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', download=download)

# Criando figura para mostrar as imagens
plt.figure(figsize=(15, 8))

# Mostrando 10 imagens aleatórias
indices = np.random.randint(0, len(train_dataset), 10)

for i, idx in enumerate(indices):
    img, label = train_dataset[idx]
    # Convertendo o tensor para numpy array
    img_np = np.array(img)
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(img_np, cmap='gray')
    plt.title(f'Label: {info["label"][str(label.item())]}')
    plt.axis('off')

plt.suptitle('Amostras do PathMNIST Dataset', fontsize=16)
plt.tight_layout()
plt.savefig('pathmnist_samples.png')
print("\nImagens salvas em 'pathmnist_samples.png'") 