from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def carregar_modelos():
    print("Carregando modelos...")
    # Modelo de análise de sentimento
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    
    # Modelo de resumo
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Modelo para RAG
    modelo_nome = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModel.from_pretrained(modelo_nome)
    
    return sentiment_analyzer, summarizer, tokenizer, modelo

def gerar_embedding(texto, tokenizer, modelo):
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = modelo(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def analise_sentimento(texto, analyzer):
    resultado = analyzer(texto)
    label_map = {'positive': 'Positivo', 'negative': 'Negativo', 'neutral': 'Neutro'}
    return f"Sentimento: {label_map.get(resultado[0]['label'], resultado[0]['label'])}\nConfiança: {resultado[0]['score']:.2f}"

def resumir_texto(texto, summarizer):
    resumo = summarizer(texto, max_length=130, min_length=30, do_sample=False)
    return resumo[0]['summary_text']

def consulta_rag(pergunta, base_conhecimento, tokenizer, modelo):
    embedding_pergunta = gerar_embedding(pergunta, tokenizer, modelo)
    embeddings_base = np.vstack([gerar_embedding(texto, tokenizer, modelo) for texto in base_conhecimento])
    similaridades = cosine_similarity(embedding_pergunta, embeddings_base)[0]
    indice_mais_similar = np.argmax(similaridades)
    return base_conhecimento[indice_mais_similar]

def main():
    # Carregando modelos
    sentiment_analyzer, summarizer, tokenizer, modelo = carregar_modelos()
    
    # Base de conhecimento para RAG
    base_conhecimento = [
        "A inteligência artificial está transformando a maneira como vivemos.",
        "Machine Learning é uma subárea da IA focada em algoritmos de aprendizado.",
        "Deep Learning utiliza redes neurais profundas para aprendizado.",
        "RAG (Retrieval Augmented Generation) combina recuperação de informações com geração de texto."
    ]
    
    while True:
        print("\n=== Sistema Integrado de IA ===")
        print("1. Análise de Sentimento")
        print("2. Resumo de Texto")
        print("3. Consulta RAG")
        print("4. Sair")
        
        opcao = input("\nEscolha uma opção (1-4): ")
        
        if opcao == "1":
            texto = input("\nDigite o texto para análise de sentimento: ")
            print("\n" + analise_sentimento(texto, sentiment_analyzer))
            
        elif opcao == "2":
            texto = input("\nDigite o texto para resumir: ")
            print("\nResumo:", resumir_texto(texto, summarizer))
            
        elif opcao == "3":
            pergunta = input("\nFaça sua pergunta: ")
            resposta = consulta_rag(pergunta, base_conhecimento, tokenizer, modelo)
            print("\nResposta mais relevante:", resposta)
            
        elif opcao == "4":
            print("\nEncerrando o programa...")
            break
            
        else:
            print("\nOpção inválida!")

if __name__ == "__main__":
    main() 