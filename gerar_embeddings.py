import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# 🚀 Carregar o dataset completo
dataset = pd.read_csv(r'C:\Users\mlzengo\Documents\TJGO\SPRINT IV\data\processed\DATASET_FINAL_TRATADO.csv')
model = SentenceTransformer('all-mpnet-base-v2')

# ✅ Função para gerar embeddings
def gerar_embedding(texto):
    return model.encode(texto)

# 🔄 Preparar os textos para gerar embeddings
embeddings = []
referencias = []

for index, row in dataset.iterrows():
    bloco = f"""
    Diagnóstico: {row['diagnóstico']}.
    Conclusão: {row['conclusão']}.
    Justificativa: {row['conclusão justificada']}.
    CID: {row['cid']}.
    Princípio Ativo: {row['princípio ativo']}.
    Nome Comercial: {row['nome comercial']}.
    Tipo da Tecnologia: {row['tipo da tecnologia']}.
    Órgão: {row['órgão']}.
    Serventia: {row['serventia']}.
    """
    embedding = gerar_embedding(bloco)
    embeddings.append(embedding)
    referencias.append(row['link visualização'])

# ✅ Salvar os embeddings e referências
with open('embeddings.pkl', 'wb') as f:
    pickle.dump((embeddings, referencias), f)

print("✅ Embeddings gerados e salvos com sucesso!")
