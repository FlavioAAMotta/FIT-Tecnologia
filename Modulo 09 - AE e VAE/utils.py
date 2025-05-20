import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def loss_function(recon_x, x, mu, log_var):
    """
    Calcula a loss do VAE (Reconstruction + KL divergence)
    """
    # Erro de reconstrução usando BCE
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD

def plot_reconstructions(model, data_loader, device, num_images=8):
    """
    Plota imagens originais e suas reconstruções
    """
    model.eval()
    with torch.no_grad():
        images = next(iter(data_loader))[0][:num_images].to(device)
        reconstructions, _, _ = model(images)
        
        # Concatena imagens originais e reconstruções
        comparison = torch.cat([images, reconstructions])
        
        plt.figure(figsize=(12, 4))
        grid = vutils.make_grid(comparison, nrow=num_images, normalize=True, padding=2)
        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)), cmap='gray')
        plt.axis('off')
        plt.title('Original (top) vs Reconstruído (bottom)')
        plt.show()

def plot_generated_samples(model, device, num_samples=64):
    """
    Plota amostras geradas pelo modelo
    """
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device)
        
        plt.figure(figsize=(8, 8))
        grid = vutils.make_grid(samples, nrow=8, normalize=True, padding=2)
        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)), cmap='gray')
        plt.axis('off')
        plt.title('Amostras Geradas')
        plt.show() 