import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

class IndexadorPDF:
    def __init__(self, pasta_pdfs="data"):
        self.pasta_pdfs = pasta_pdfs
        self.modelo = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configurar ChromaDB
        self.cliente = chromadb.Client()
        self.colecao = self.cliente.create_collection(
            name="pdfs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        )
    
    def extrair_texto_pdf(self, caminho_arquivo):
        """Extrai texto de um arquivo PDF."""
        documento = fitz.open(caminho_arquivo)
        texto_completo = []
        
        for pagina in documento:
            texto = pagina.get_text()
            # Dividir em parágrafos
            paragrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]
            texto_completo.extend(paragrafos)
        
        return texto_completo
    
    def indexar_pdfs(self):
        """Indexa todos os PDFs na pasta especificada."""
        for arquivo in os.listdir(self.pasta_pdfs):
            if arquivo.endswith('.pdf'):
                caminho_completo = os.path.join(self.pasta_pdfs, arquivo)
                print(f"Indexando {arquivo}...")
                
                # Extrair texto
                paragrafos = self.extrair_texto_pdf(caminho_completo)
                
                # Adicionar ao ChromaDB
                ids = [f"{arquivo}_{i}" for i in range(len(paragrafos))]
                metadados = [{"fonte": arquivo, "pagina": i//10 + 1} for i in range(len(paragrafos))]
                
                self.colecao.add(
                    documents=paragrafos,
                    ids=ids,
                    metadatas=metadados
                )
                
                print(f"✓ {len(paragrafos)} parágrafos indexados de {arquivo}")
    
    def buscar_similar(self, consulta, n_resultados=3):
        """Busca trechos similares à consulta."""
        resultados = self.colecao.query(
            query_texts=[consulta],
            n_results=n_resultados
        )
        
        return resultados

if __name__ == "__main__":
    # Teste do indexador
    indexador = IndexadorPDF()
    indexador.indexar_pdfs() 