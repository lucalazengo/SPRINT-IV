import os
import pandas as pd
import fitz  
import re
from tqdm import tqdm

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(base_dir, "../data/pdfs_notas_tecnicas_goias")
output_csv = os.path.join(base_dir, "../data/processed/web_natjus_consolidado.csv")

# Estrutura para armazenar os dados
data_extracao = []

# Expressões Regulares para Extração
patterns = {
    "ID": r"Nota Técnica (\d+)",
    "Data de Conclusão": r"Data de conclusão:\s*(.*)",
    "Idade": r"Idade:\s*(\d+)",
    "Sexo": r"Sexo:\s*(\w+)",
    "Cidade": r"Cidade:\s*(.*?)(?:/|\n)",
    "Órgão": r"Esfera/Órgão:\s*(.*)",
    "Serventia": r"Vara/Serventia:\s*(.*)",
    "CID": r"CID:\s*(.*)",
    "Diagnóstico": r"Diagnóstico:\s*(.*)",
    "Tipo da Tecnologia": r"Tipo da Tecnologia:\s*(.*)",
    "Princípio Ativo": r"Princípio Ativo:\s*(.*)",
    "Posologia": r"Posologia:\s*(.*)",
    "Descrição": r"Descrição:\s*(.*)",
    "Procedimento SUS": r"O procedimento está inserido no SUS\?\s*(.*)",
    "Evidências Científicas": r"Há evidências científicas\?\s*(.*)",
    "Princípio Ativo": r"Princípio Ativo:\s*(.*)",
    "Nome Comercial": r"Nome comercial:\s*(.*)",
    "Tecnologia Disponível": r"Tecnologia:\s*(.*)",
    "Custo da Tecnologia": r"Custo da tecnologia:\s*(.*)",
    "Conclusão Justificada": r"Conclusão Justificada:\s*(.*)",
    "Conclusão": r"Conclusão:\s*(.*)",
    "NatJus Responsável": r"NatJus Responsável:\s*(.*)",
    "Instituição Responsável:": r"Instituição Responsável:\s*(.*)"
}

# Função para extrair dados do PDF
def extrair_dados_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        texto = ""
        for page in pdf:
            texto += page.get_text()

        # Aplicar expressões regulares
        extracao = {"Arquivo": os.path.basename(pdf_path)}
        for key, pattern in patterns.items():
            resultado = re.search(pattern, texto, re.MULTILINE)
            extracao[key] = resultado.group(1) if resultado else None

        return extracao

# Leitura dos PDFs da pasta
def processar_pdfs():
    for pdf_file in tqdm(os.listdir(pdf_dir), desc="Processando PDFs"):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            dados = extrair_dados_pdf(pdf_path)
            data_extracao.append(dados)

# Gerar o DataFrame e salvar
def salvar_csv():
    df = pd.DataFrame(data_extracao)
    df.to_csv(output_csv, index=False)
    print(f" Arquivo CSV gerado em: {output_csv}")

if __name__ == "__main__":
    processar_pdfs()
    salvar_csv()
