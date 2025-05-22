# Mini RAG - Sistema de Consulta de Notas de Aula

Este é um sistema simples de RAG (Retrieval-Augmented Generation) que permite fazer consultas sobre notas de aula em formato texto. A versão atual usa apenas processamento local, sem necessidade de APIs externas.

## Requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)

## Instalação

1. Clone este repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como usar

1. Execute o script:
```bash
python mini_rag_simples.py
```

2. Digite sua pergunta sobre o conteúdo das notas de aula
3. O sistema responderá com a sentença mais relevante do texto
4. Digite 'sair' para encerrar o programa

## Funcionalidades

- Processamento local do texto
- Busca por similaridade usando TF-IDF
- Respostas baseadas no contexto das notas
- Não requer conexão com internet ou APIs externas

## Exemplo de Uso

```
Sistema de Consulta de Notas de Aula
Digite 'sair' para encerrar

Sua pergunta: O que é aprendizado supervisionado?

Resposta: O aprendizado supervisionado é um tipo de aprendizado de máquina onde o algoritmo aprende a partir de exemplos rotulados. 