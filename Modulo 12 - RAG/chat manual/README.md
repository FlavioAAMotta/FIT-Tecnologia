# Chat sobre o Manual do Aluno

Este é um sistema RAG (Retrieval-Augmented Generation) que permite fazer consultas sobre o manual do aluno, com suporte a diferentes formatos de arquivo (PDF, DOCX, HTML) e citações das fontes.

## Características

- Processamento de múltiplos formatos de arquivo (PDF, DOCX, HTML)
- Manutenção de metadados (páginas, seções, parágrafos)
- Respostas com citações das fontes
- Interface de linha de comando simples
- Suporte a português

## Requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)

## Instalação

1. Clone este repositório
2. Crie um ambiente virtual:
```bash
python -m venv venv
```

3. Ative o ambiente virtual:
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Estrutura de Arquivos

```
projeto2/
├── manual/           # Pasta para os arquivos do manual
├── manual_rag.py     # Código principal
├── requirements.txt  # Dependências
└── README.md        # Este arquivo
```

## Como usar

1. Coloque os arquivos do manual (PDF, DOCX, HTML) na pasta `manual/`

2. Execute o script:
```bash
python manual_rag.py
```

3. Faça suas perguntas sobre o manual

4. O sistema responderá com:
   - Uma resposta natural baseada no conteúdo
   - Referências das fontes utilizadas
   - Pontuação de similaridade para cada referência

## Exemplo de Uso

```
Sistema de Consulta ao Manual do Aluno
Digite 'sair' para encerrar

Sua pergunta: Quais são os requisitos para matrícula?

Resposta: De acordo com o manual, os requisitos para matrícula incluem...

Referências utilizadas:
- manual_matricula.pdf, página 5 (similaridade: 0.89)
- manual_graduacao.pdf, página 12 (similaridade: 0.75)
```

## Notas

- O sistema mantém o contexto das páginas/seções para referência
- As respostas são geradas usando um modelo de linguagem em português
- A similaridade indica quão relevante é cada trecho para a pergunta 