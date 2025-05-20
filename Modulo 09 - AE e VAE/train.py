import torch
from torch.optim import Adam
from tqdm import tqdm
from utils import loss_function, plot_reconstructions, plot_generated_samples

def train_vae(model, train_loader, device, num_epochs=5, learning_rate=1e-3):
    """
    Treina o VAE
    """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, (data, _) in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
            progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}')
            progress_bar.set_postfix({'Loss': loss.item() / len(data)})
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}')
        
        # Visualiza resultados a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            plot_reconstructions(model, train_loader, device)
            plot_generated_samples(model, device)
    
    return model 