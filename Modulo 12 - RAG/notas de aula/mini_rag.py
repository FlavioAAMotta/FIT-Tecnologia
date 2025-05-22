import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(page_title="Mini RAG - Notas de Aula", page_icon="📚")
st.title("📚 Mini RAG - Sistema de Consulta de Notas de Aula")

# Inicialização do modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Função para processar o arquivo de texto
def process_text(text):
    # Dividir o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Criar banco de dados vetorial
    vectorstore = Chroma.from_texts(chunks, embeddings)
    
    return vectorstore

# Interface do usuário
uploaded_file = st.file_uploader("Faça upload do arquivo de notas (.txt)", type=['txt'])

if uploaded_file is not None:
    # Ler o conteúdo do arquivo
    text = uploaded_file.read().decode('utf-8')
    
    # Processar o texto
    vectorstore = process_text(text)
    
    # Criar o sistema de perguntas e respostas
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5}
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Campo para a pergunta do usuário
    user_question = st.text_input("Faça sua pergunta sobre as notas de aula:")
    
    if user_question:
        # Obter resposta
        response = qa_chain.run(user_question)
        
        # Exibir resposta
        st.write("Resposta:")
        st.write(response) 