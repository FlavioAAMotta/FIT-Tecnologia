import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Configurando o dispositivo (GPU se disponível)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# Definindo hiperparâmetros
latent_dim = 100
image_size = 28
batch_size = 32
num_epochs = 100
lr = 0.0002
beta1 = 0.5

# Carregando o dataset PathMNIST
data_flag = 'pathmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

# Definindo as transformações
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregando o dataset com as transformações
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', transform=transform, download=download)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Definindo o Gerador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Camada de entrada
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            
            # Camadas intermediárias
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            
            # Camada de saída
            nn.Linear(1024, image_size * image_size * n_channels),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.main(x)
        img = img.view(-1, n_channels, image_size, image_size)
        return img

# Definindo o Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Camada de entrada
            nn.Linear(image_size * image_size * n_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Camadas intermediárias
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Camada de saída
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, image_size * image_size * n_channels)
        return self.main(x)

# Inicializando os modelos
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Definindo as funções de perda e otimizadores
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Função para salvar imagens geradas
def save_generated_images(epoch, generator, fixed_noise):
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(fake_images[i][0].numpy(), cmap='gray')
            plt.axis('off')
        plt.savefig(f'fake_images_epoch_{epoch}.png')
        plt.close()

# Gerando ruído fixo para visualização
fixed_noise = torch.randn(16, latent_dim).to(device)

# Loop de treinamento
print("Iniciando treinamento...")

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Labels para dados reais e falsos
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Treinar o Discriminador
        discriminator.zero_grad()
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Treinar o Gerador
        generator.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
    
    # Salvar imagens a cada 10 épocas
    if (epoch + 1) % 10 == 0:
        save_generated_images(epoch + 1, generator, fixed_noise)

print("Treinamento concluído!")

# Salvar os modelos
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth') 