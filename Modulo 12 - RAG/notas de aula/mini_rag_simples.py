from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def carregar_notas(caminho_arquivo):
    """Carrega o arquivo de notas e retorna o texto."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        return arquivo.read()

def dividir_em_paragrafos(texto):
    """Divide o texto em parágrafos significativos."""
    # Remove linhas vazias e divide por quebras de linha duplas
    paragrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
    return paragrafos

def gerar_embedding(texto, tokenizer, modelo):
    """Gera embedding para um texto usando o modelo BERT."""
    # Tokeniza o texto
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Gera os embeddings
    with torch.no_grad():
        outputs = modelo(**inputs)
    
    # Usa a média dos embeddings da última camada como representação do texto
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.numpy()

def encontrar_resposta(pergunta, paragrafos, tokenizer, modelo):
    """Encontra a resposta mais relevante para a pergunta usando embeddings."""
    # Gera embeddings para a pergunta e todos os parágrafos
    embedding_pergunta = gerar_embedding(pergunta, tokenizer, modelo)
    embeddings_paragrafos = np.vstack([gerar_embedding(p, tokenizer, modelo) for p in paragrafos])
    
    # Calcula similaridade entre a pergunta e os parágrafos
    similaridades = cosine_similarity(embedding_pergunta, embeddings_paragrafos)[0]
    
    # Encontra o parágrafo mais similar
    indice_mais_similar = np.argmax(similaridades)
    
    return paragrafos[indice_mais_similar]

def main():
    print("Carregando modelo de linguagem...")
    # Carrega o modelo e tokenizer em português
    modelo_nome = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModel.from_pretrained(modelo_nome)
    
    print("Carregando notas de aula...")
    # Carrega as notas
    texto = carregar_notas('notas_aula.txt')
    paragrafos = dividir_em_paragrafos(texto)
    
    print("\nSistema de Consulta de Notas de Aula")
    print("Digite 'sair' para encerrar")
    
    while True:
        pergunta = input("\nSua pergunta: ")
        if pergunta.lower() == 'sair':
            break
            
        resposta = encontrar_resposta(pergunta, paragrafos, tokenizer, modelo)
        print("\nResposta:", resposta)

if __name__ == "__main__":
    main() 