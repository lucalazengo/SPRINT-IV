import os
import requests
import pandas as pd
from tqdm import tqdm

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "pdfs_notas_tecnicas_goias")
csv_path = os.path.join(base_dir, "notas_tecnicas_goias.csv")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def download_pdfs():
    """
    Realiza o download dos PDFs listados no CSV de notas técnicas.
    """
    # Lê o arquivo CSV gerado anteriormente
    df = pd.read_csv(csv_path)

    # Itera sobre as linhas para baixar os PDFs
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Baixando PDFs"):
        pdf_url = row["Link Download"]
        nota_id = row["ID"]

        # Define o caminho para salvar o PDF
        pdf_path = os.path.join(output_dir, f"{nota_id}.pdf")
        
        # Verifica se o arquivo já existe para não baixar novamente
        if not os.path.exists(pdf_path):
            try:
                response = requests.get(pdf_url, stream=True)
                if response.status_code == 200:
                    with open(pdf_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    print(f" Download concluído: {nota_id}.pdf")
                else:
                    print(f" Falha no download: {nota_id}.pdf - Status {response.status_code}")
            except Exception as e:
                print(f" Erro no download {nota_id}.pdf: {e}")
        else:
            print(f" Arquivo já existe: {nota_id}.pdf")

if __name__ == "__main__":
    download_pdfs()
