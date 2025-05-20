import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import VAE
from train import train_vae
from utils import plot_reconstructions, plot_generated_samples

def main():
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    latent_dim = 32
    num_epochs = 4
    
    # Transformações para as imagens
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Carrega o dataset Fashion-MNIST
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Inicializa o modelo
    model = VAE(latent_dim=latent_dim).to(device)
    print("Modelo criado com sucesso!")
    
    # Treina o modelo
    print("Iniciando treinamento...")
    model = train_vae(model, train_loader, device, num_epochs=num_epochs)
    
    # Gera e plota algumas amostras finais
    print("Gerando amostras finais...")
    plot_reconstructions(model, train_loader, device)
    plot_generated_samples(model, device)
    
    # Salva o modelo treinado
    torch.save(model.state_dict(), 'vae_fashion_mnist.pth')
    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    main() 