# 📚 Pergunte ao Paper

Um aplicativo que permite fazer perguntas sobre artigos científicos em PDF e receber respostas ilustradas com imagens geradas por IA.

## 🚀 Funcionalidades

- Upload e processamento de PDFs
- Busca semântica no conteúdo dos artigos
- Geração de imagens ilustrativas usando Stable Diffusion
- Interface web amigável com Streamlit

## 📋 Requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)
- GPU com CUDA (recomendado para geração de imagens)

## 🛠️ Instalação

1. Clone este repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd projeto-rag-gen
```

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

## 🎮 Como Usar

1. Inicie o aplicativo:
```bash
streamlit run app.py
```

2. Acesse a interface web no navegador (geralmente em http://localhost:8501)

3. Faça upload dos PDFs que deseja consultar

4. Digite sua pergunta sobre o conteúdo dos artigos

5. O sistema irá:
   - Buscar trechos relevantes nos PDFs
   - Gerar imagens ilustrativas para cada trecho
   - Exibir os resultados com as imagens

## 📁 Estrutura do Projeto

```
projeto-rag-gen/
├── app.py               # Interface Streamlit
├── indexador.py         # Processamento e indexação de PDFs
├── generator.py         # Geração de imagens
├── requirements.txt     # Dependências
├── data/               # Pasta para PDFs
└── outputs/            # Pasta para imagens geradas
```

## 🔧 Tecnologias Utilizadas

- **Streamlit**: Interface web
- **ChromaDB**: Base de dados vetorial
- **Sentence Transformers**: Embeddings de texto
- **PyMuPDF**: Processamento de PDFs
- **Stable Diffusion**: Geração de imagens

## 📝 Notas

- A primeira execução pode demorar mais tempo devido ao download dos modelos
- A geração de imagens é mais rápida com GPU
- Os PDFs são processados e indexados localmente

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes. 