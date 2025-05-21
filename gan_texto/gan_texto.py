import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações
SEQ_LENGTH = 20
BATCH_SIZE = 32
NUM_EPOCHS = 200
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
LEARNING_RATE = 0.0001
TEMPERATURE = 1.0
LABEL_SMOOTHING = 0.1
NOISE_FACTOR = 0.1

#Dataset de versos reais
VERSOS = [
    "No meio do caminho tinha uma pedra",
    "Tinha uma pedra no meio do caminho",
    "Tinha uma pedra no meio do caminho tinha uma pedra",
    "No meio do caminho tinha uma pedra tinha uma pedra",
    "No meio do caminho tinha uma pedra no meio do caminho",
    "A pedra que no meio do caminho tinha",
    "No meio do caminho tinha uma pedra que tinha",
    "Tinha uma pedra que no meio do caminho tinha",
    "No meio do caminho tinha uma pedra que no meio",
    "Tinha uma pedra que tinha no meio do caminho",
    "No meio do caminho tinha uma pedra que tinha no meio",
    "Tinha uma pedra que tinha no meio do caminho tinha",
    "No meio do caminho tinha uma pedra que tinha no meio do caminho",
    "Tinha uma pedra que tinha no meio do caminho tinha uma pedra",
    "No meio do caminho tinha uma pedra que tinha no meio do caminho tinha uma pedra",
    "A pedra que tinha no meio do caminho tinha uma pedra",
    "No meio do caminho tinha uma pedra que tinha uma pedra",
    "Tinha uma pedra que tinha uma pedra no meio do caminho",
    "No meio do caminho tinha uma pedra que tinha uma pedra no meio",
    "Tinha uma pedra que tinha uma pedra no meio do caminho tinha"
]

class Vocabulario:
    def __init__(self, versos):
        # Criar vocabulário a partir dos versos
        palavras = []
        for verso in versos:
            palavras.extend(verso.lower().split())
        
        # Contar frequência das palavras
        contador = Counter(palavras)
        
        # Criar dicionários de mapeamento
        self.palavra_para_idx = {palavra: idx for idx, (palavra, _) in enumerate(contador.most_common())}
        self.idx_para_palavra = {idx: palavra for palavra, idx in self.palavra_para_idx.items()}
        self.vocab_size = len(self.palavra_para_idx)
        
    def texto_para_indices(self, texto):
        palavras = texto.lower().split()
        return [self.palavra_para_idx.get(palavra, 0) for palavra in palavras]
    
    def indices_para_texto(self, indices):
        return ' '.join([self.idx_para_palavra.get(idx, '<UNK>') for idx in indices])

class VersosDataset(Dataset):
    def __init__(self, versos, vocab):
        self.vocab = vocab
        self.data = []
        
        # Converter versos para índices
        for verso in versos:
            indices = vocab.texto_para_indices(verso)
            # Padding ou truncamento para SEQ_LENGTH
            if len(indices) < SEQ_LENGTH:
                indices.extend([0] * (SEQ_LENGTH - len(indices)))
            else:
                indices = indices[:SEQ_LENGTH]
            self.data.append(indices)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])

# Inicialização adequada de pesos
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)

# Gerador
class Generator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)
        self.dropout = nn.Dropout(0.3)
        self.apply(init_weights)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedded)
        logits = self.fc(self.dropout(output))
        return logits

    def sample(self, x, temperature=1.0):
        logits = self.forward(x)
        probs = torch.softmax(logits / temperature, dim=-1)
        # Reshape para 2D: (batch_size * seq_length, vocab_size)
        probs_2d = probs.reshape(-1, probs.size(-1))
        # Amostrar índices
        samples = torch.multinomial(probs_2d, num_samples=1)
        # Reshape de volta para (batch_size, seq_length)
        return samples.reshape(x.size())

# Discriminador
class Discriminator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        self.dropout = nn.Dropout(0.3)
        self.apply(init_weights)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedded)
        return torch.sigmoid(self.fc(output[:, -1, :]))

# Treinamento com melhorias
def train_gan():
    vocab = Vocabulario(VERSOS)
    generator = Generator(vocab.vocab_size).to(device)
    discriminator = Discriminator(vocab.vocab_size).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    dataset = VersosDataset(VERSOS, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    lr_scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, 'min', patience=5)
    lr_scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, 'min', patience=5)

    for epoch in range(NUM_EPOCHS):
        generator.train()
        discriminator.train()

        epoch_d_loss, epoch_g_loss = 0, 0

        for real_data in tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            real_label = torch.ones(batch_size, 1, device=device) * (1 - LABEL_SMOOTHING)
            fake_label = torch.zeros(batch_size, 1, device=device) + LABEL_SMOOTHING

            # Treinar Discriminador
            d_optimizer.zero_grad()

            # Ruído mais realista nos dados reais
            real_data_noisy = real_data.clone()
            mask = torch.rand_like(real_data_noisy.float()) < NOISE_FACTOR
            real_data_noisy[mask] = torch.randint(0, vocab.vocab_size, (mask.sum(),)).to(device)

            output_real = discriminator(real_data_noisy)
            d_loss_real = criterion(output_real, real_label)

            noise = torch.randint(0, vocab.vocab_size, (batch_size, SEQ_LENGTH), device=device)
            fake_data = generator.sample(noise, TEMPERATURE)
            output_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(output_fake, fake_label)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Treinar Gerador
            g_optimizer.zero_grad()
            fake_data = generator.sample(noise, TEMPERATURE)
            output_fake = discriminator(fake_data)
            g_loss = criterion(output_fake, real_label)

            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        # Ajuste automático da taxa de aprendizado
        lr_scheduler_g.step(epoch_g_loss)
        lr_scheduler_d.step(epoch_d_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] d_loss: {epoch_d_loss/len(dataloader):.4f}, g_loss: {epoch_g_loss/len(dataloader):.4f}')
            generator.eval()
            with torch.no_grad():
                noise = torch.randint(0, vocab.vocab_size, (3, SEQ_LENGTH), device=device)
                generated = generator.sample(noise, TEMPERATURE)
                print("\nVersos gerados:")
                for verso in generated:
                    texto = vocab.indices_para_texto(verso.cpu().numpy())
                    print(texto)

if __name__ == "__main__":
    train_gan()
