import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup

class ManualProcessor:
    def __init__(self):
        # Carregar modelo BERT em português
        self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        
        self.segmentos = []
        self.metadados = []
        
    def processar_pdf(self, arquivo):
        with open(arquivo, 'rb') as f:
            leitor = PyPDF2.PdfReader(f)
            for pagina in range(len(leitor.pages)):
                texto = leitor.pages[pagina].extract_text()
                paragrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
                
                for i, paragrafo in enumerate(paragrafos):
                    self.segmentos.append(paragrafo)
                    self.metadados.append({
                        'arquivo': os.path.basename(arquivo),
                        'pagina': pagina + 1,
                        'paragrafo': i + 1
                    })
    
    def processar_docx(self, arquivo):
        doc = Document(arquivo)
        for i, paragrafo in enumerate(doc.paragraphs):
            if paragrafo.text.strip():
                self.segmentos.append(paragrafo.text.strip())
                self.metadados.append({
                    'arquivo': os.path.basename(arquivo),
                    'secao': paragrafo.style.name,
                    'paragrafo': i + 1
                })
    
    def processar_html(self, arquivo):
        with open(arquivo, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            for i, paragrafo in enumerate(soup.find_all(['p', 'h1', 'h2', 'h3'])):
                if paragrafo.text.strip():
                    self.segmentos.append(paragrafo.text.strip())
                    self.metadados.append({
                        'arquivo': os.path.basename(arquivo),
                        'tag': paragrafo.name,
                        'paragrafo': i + 1
                    })
    
    def gerar_embedding(self, texto):
        inputs = self.tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def encontrar_segmentos_relevantes(self, pergunta, top_k=3):
        embedding_pergunta = self.gerar_embedding(pergunta)
        embeddings_segmentos = np.array([self.gerar_embedding(seg)[0] for seg in self.segmentos])
        
        similaridades = cosine_similarity(embedding_pergunta, embeddings_segmentos)[0]
        indices_top = np.argsort(similaridades)[-top_k:][::-1]
        
        return [(self.segmentos[i], self.metadados[i], similaridades[i]) for i in indices_top]
    
    def gerar_resposta(self, pergunta, segmentos_relevantes):
        # Criar uma resposta estruturada com os trechos relevantes
        resposta = []
        resposta.append("Com base no manual, encontrei as seguintes informações relevantes:")
        
        for i, (seg, meta, sim) in enumerate(segmentos_relevantes, 1):
            if 'pagina' in meta:
                fonte = f"{meta['arquivo']}, página {meta['pagina']}"
            elif 'secao' in meta:
                fonte = f"{meta['arquivo']}, seção {meta['secao']}"
            else:
                fonte = f"{meta['arquivo']}, {meta['tag']} {meta['paragrafo']}"
            
            resposta.append(f"\n{i}. {seg}")
            resposta.append(f"   Fonte: {fonte}")
        
        return "\n".join(resposta)

def main():
    processor = ManualProcessor()
    
    # Processar arquivos na pasta manual
    pasta_manual = "manual"
    for arquivo in os.listdir(pasta_manual):
        caminho = os.path.join(pasta_manual, arquivo)
        if arquivo.endswith('.pdf'):
            processor.processar_pdf(caminho)
        elif arquivo.endswith('.docx'):
            processor.processar_docx(caminho)
        elif arquivo.endswith('.html'):
            processor.processar_html(caminho)
    
    print("Sistema de Consulta ao Manual do Aluno")
    print("Digite 'sair' para encerrar\n")
    
    while True:
        pergunta = input("Sua pergunta: ").strip()
        if pergunta.lower() == 'sair':
            break
        
        segmentos_relevantes = processor.encontrar_segmentos_relevantes(pergunta)
        resposta = processor.gerar_resposta(pergunta, segmentos_relevantes)
        
        print("\nResposta:", resposta)
        print("\nReferências utilizadas:")
        for seg, meta, sim in segmentos_relevantes:
            if 'pagina' in meta:
                print(f"- {meta['arquivo']}, página {meta['pagina']} (similaridade: {sim:.2f})")
            elif 'secao' in meta:
                print(f"- {meta['arquivo']}, seção {meta['secao']} (similaridade: {sim:.2f})")
            else:
                print(f"- {meta['arquivo']}, {meta['tag']} {meta['paragrafo']} (similaridade: {sim:.2f})")
        print()

if __name__ == "__main__":
    main() 