# Variational Autoencoder (VAE) para Fashion-MNIST

Este projeto implementa um Variational Autoencoder (VAE) para o dataset Fashion-MNIST usando PyTorch. O VAE é capaz de gerar novas imagens de roupas e acessórios, além de reconstruir imagens existentes.

## Estrutura do Projeto

- `model.py`: Implementação da arquitetura do VAE
- `train.py`: Funções de treinamento
- `utils.py`: Funções auxiliares para loss e visualização
- `main.py`: Script principal para executar o treinamento
- `requirements.txt`: Dependências do projeto

## Requisitos

Para instalar as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Como Usar

1. Clone o repositório
2. Instale as dependências
3. Execute o script principal:

```bash
python main.py
```

## Funcionalidades

- Treinamento do VAE no dataset Fashion-MNIST
- Visualização de reconstruções durante o treinamento
- Geração de novas imagens a partir do espaço latente
- Salvamento do modelo treinado

## Arquitetura do VAE

O VAE implementado possui:
- Encoder convolucional
- Espaço latente de 32 dimensões
- Decoder convolucional
- Função de perda que combina reconstrução (BCE) e regularização (KL divergence)

## Resultados

Durante o treinamento, você verá:
- Progresso do treinamento com loss por época
- Visualizações de reconstruções a cada 10 épocas
- Amostras geradas a partir do espaço latente

O modelo treinado será salvo como `vae_fashion_mnist.pth`. 