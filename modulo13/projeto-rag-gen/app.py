import streamlit as st
import os
from indexador import IndexadorPDF
from generator import GeradorImagem

# Configuração da página
st.set_page_config(
    page_title="Pergunte ao Paper",
    page_icon="📚",
    layout="wide"
)

# Título e descrição
st.title("📚 Pergunte ao Paper")
st.markdown("""
Este aplicativo permite que você faça perguntas sobre artigos científicos em PDF
e receba respostas ilustradas com imagens geradas por IA.
""")

# Inicializar componentes
@st.cache_resource
def inicializar_componentes():
    return IndexadorPDF(), GeradorImagem()

indexador, gerador = inicializar_componentes()

# Upload de PDFs
st.header("1. Upload de PDFs")
pdfs = st.file_uploader("Selecione os arquivos PDF", type="pdf", accept_multiple_files=True)

if pdfs:
    # Criar pasta data se não existir
    os.makedirs("data", exist_ok=True)
    
    # Salvar PDFs
    for pdf in pdfs:
        with open(os.path.join("data", pdf.name), "wb") as f:
            f.write(pdf.getvalue())
    
    # Indexar PDFs
    with st.spinner("Indexando PDFs..."):
        indexador.indexar_pdfs()
    st.success("PDFs indexados com sucesso!")

# Consulta
st.header("2. Faça sua pergunta")
pergunta = st.text_input("Digite sua pergunta sobre os artigos:")

if pergunta:
    # Buscar resposta
    with st.spinner("Buscando resposta..."):
        resultados = indexador.buscar_similar(pergunta)
    
    # Exibir resultados
    st.header("3. Resultados")
    
    for i, (doc, meta) in enumerate(zip(resultados['documents'][0], resultados['metadatas'][0])):
        with st.expander(f"Trecho {i+1} - {meta['fonte']} (Página {meta['pagina']})"):
            st.write(doc)
            
            # Gerar imagem
            with st.spinner("Gerando imagem..."):
                caminho_imagem = gerador.gerar_imagem(doc)
                st.image(caminho_imagem, caption="Ilustração gerada")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit, ChromaDB e Stable Diffusion") 