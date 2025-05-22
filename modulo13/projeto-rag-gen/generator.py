import torch
from diffusers import StableDiffusionPipeline
import os

class GeradorImagem:
    def __init__(self, pasta_saida="outputs"):
        self.pasta_saida = pasta_saida
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
    
    def criar_prompt(self, texto):
        """Cria um prompt para geração de imagem baseado no texto."""
        # Simplificar o texto para criar um prompt mais direto
        palavras_chave = texto.split()[:10]  # Pegar as primeiras 10 palavras
        prompt = " ".join(palavras_chave)
        
        # Adicionar contexto para melhorar a qualidade da imagem
        prompt = f"high quality illustration of {prompt}, detailed, professional, educational"
        
        return prompt
    
    def gerar_imagem(self, texto, nome_arquivo=None):
        """Gera uma imagem baseada no texto."""
        prompt = self.criar_prompt(texto)
        
        # Gerar a imagem
        imagem = self.pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        # Salvar a imagem
        if nome_arquivo is None:
            nome_arquivo = f"imagem_{hash(texto)}.png"
        
        caminho_completo = os.path.join(self.pasta_saida, nome_arquivo)
        imagem.save(caminho_completo)
        
        return caminho_completo

if __name__ == "__main__":
    # Teste do gerador
    gerador = GeradorImagem()
    caminho = gerador.gerar_imagem("Uma rede neural convolucional processando uma imagem")
    print(f"Imagem gerada: {caminho}") 