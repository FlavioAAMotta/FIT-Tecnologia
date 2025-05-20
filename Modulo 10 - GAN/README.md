# Módulo 10 - Geração de Dados Médicos Sintéticos com GANs

## 📋 Descrição
Este módulo implementa uma Rede Adversária Generativa (GAN) para gerar imagens médicas sintéticas usando o dataset PathMNIST. O objetivo é demonstrar como GANs podem ser utilizadas para aumentar datasets médicos e auxiliar no treinamento de modelos de diagnóstico.

## 🎯 Objetivo
- Gerar imagens médicas sintéticas realistas
- Aumentar a disponibilidade de dados para treinamento
- Demonstrar o uso de GANs em contexto médico

## 🔧 Requisitos
```
torch>=1.9.0
torchvision>=0.10.0
medmnist>=2.2.3
numpy>=1.21.0
matplotlib>=3.4.3
pandas>=1.3.0
scikit-learn>=0.24.2
```

## 💻 Estrutura do Projeto
- `medmnist_gan.py`: Implementação principal da GAN
- `visualize_medmnist.py`: Script para visualização do dataset
- `requirements.txt`: Dependências do projeto

## 🚀 Como Usar

### 1. Instalação
```bash
# Instalar dependências
pip install -r requirements.txt

# Instalar certificados (se necessário)
/Applications/Python\ 3.13/Install\ Certificates.command
```

### 2. Dataset
O projeto utiliza o PathMNIST, que contém 9 classes diferentes de tecidos:
- Adipose (tecido adiposo)
- Background (fundo)
- Debris (detritos)
- Lymphocytes (linfócitos)
- Mucus (muco)
- Smooth muscle (músculo liso)
- Normal colon mucosa (mucosa normal do cólon)
- Cancer-associated stroma (estroma associado ao câncer)
- Colorectal adenocarcinoma epithelium (epitélio de adenocarcinoma colorretal)

### 3. Treinamento
```python
# Executar treinamento da GAN
python medmnist_gan.py
```

### 4. Visualização
```python
# Visualizar amostras do dataset
python visualize_medmnist.py
```

### 5. Geração de Imagens
```python
# Após treinar o modelo
noise = torch.randn(1, latent_dim).to(device)
fake_image = generator(noise)

# Visualizar imagem gerada
fake_image = fake_image.cpu().detach().numpy()
fake_image = fake_image[0, 0]
fake_image = (fake_image + 1) / 2.0

plt.imshow(fake_image, cmap='gray')
plt.axis('off')
plt.show()
```

## 🏗️ Arquitetura

### Gerador
- Entrada: Vetor de ruído aleatório (latent_dim=100)
- Camadas densamente conectadas com LeakyReLU
- Saída: Imagem 28x28 pixels

### Discriminador
- Entrada: Imagem 28x28 pixels
- Camadas densamente conectadas com Dropout
- Saída: Probabilidade da imagem ser real (0-1)

## 📊 Resultados
- O modelo gera imagens médicas sintéticas de 28x28 pixels
- As imagens geradas podem ser usadas para:
  - Aumento de dataset
  - Treinamento de modelos de diagnóstico
  - Pesquisa médica
  - Educação médica

## 🔍 Uso Prático
1. **Pesquisa Médica**
   - Geração de casos hipotéticos
   - Estudo de variações de doenças
   - Teste de algoritmos de diagnóstico

2. **Educação**
   - Treinamento de estudantes
   - Simulação de casos raros
   - Material didático

3. **Desenvolvimento de IA**
   - Aumento de datasets
   - Balanceamento de classes
   - Validação de modelos

## ⚠️ Limitações e Considerações
- As imagens geradas são sintéticas e não devem ser usadas para diagnóstico real
- O modelo atual trabalha apenas com imagens em escala de cinza
- Requer validação por especialistas médicos

## 📚 Referências
- [MedMNIST Dataset](https://medmnist.com/)
- [GAN Paper Original](https://arxiv.org/abs/1406.2661)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📄 Licença
Este projeto é para fins educacionais. 