# GAN para Geração de Texto

Este projeto implementa uma GAN (Generative Adversarial Network) para geração de texto, especificamente focada em gerar pequenos versos ou poemas.

## Estrutura do Projeto

- `gan_texto.py`: Implementação principal da GAN
- `requirements.txt`: Dependências do projeto

## Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o script principal:
```bash
python gan_texto.py
```

## Arquitetura

O projeto implementa uma GAN com as seguintes características:

- **Gerador**: Utiliza uma rede LSTM para gerar sequências de texto
- **Discriminador**: Também utiliza LSTM para classificar se o texto é real ou gerado
- **Dataset**: Implementa um dataset simples de versos para treinamento

## Parâmetros Configuráveis

- `VOCAB_SIZE`: Tamanho do vocabulário (1000)
- `EMBEDDING_DIM`: Dimensão do embedding (256)
- `HIDDEN_DIM`: Dimensão da camada oculta (512)
- `SEQ_LENGTH`: Comprimento da sequência (20)
- `BATCH_SIZE`: Tamanho do batch (32)
- `NUM_EPOCHS`: Número de épocas de treinamento (100)

## Observações

Este é um exemplo didático de GAN para texto. Para resultados mais práticos, você pode:

1. Usar um dataset real de poemas ou textos
2. Implementar um vocabulário real com tokenização adequada
3. Ajustar os hiperparâmetros para melhor performance
4. Adicionar técnicas de regularização 