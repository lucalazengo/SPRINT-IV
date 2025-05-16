import pandas as pd

#  Carregar o dataset local
dataset = pd.read_csv('./data/processed/DATASET_FINAL_EMBEDDINGS.csv')

def buscar_referencia(texto):
    """
    Realiza uma busca no CSV para encontrar referências relacionadas ao tema.
    """
    resultados = dataset[dataset['diagnóstico'].str.contains(texto, case=False, na=False)]
    links = resultados['link visualização'].tolist()
    return links if links else ["Nenhuma referência encontrada."]
