import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

#  Carregar o modelo e os dados
model = SentenceTransformer('all-mpnet-base-v2')
dataset = pd.read_csv('./data/processed/DATASET_FINAL_EMBEDDINGS.csv')

# Carregar os embeddings e criar o índice
with open('./data/processed/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

#  Corrigido: Nome alterado para evitar conflito com Flask
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

#  Índice atualizado para evitar conflito de nome
faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

print(f" Índice de busca criado com {faiss_index.ntotal} jurisprudências indexadas.")

#  Função de busca semântica
def buscar_jurisprudencia(query, top_k=5):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
    
    resultados = []
    for idx, score in zip(indices[0], distances[0]):
        resultado = {
            'diagnóstico': dataset.loc[idx, 'diagnóstico'],
            'conclusão': dataset.loc[idx, 'conclusão'],
            'conclusão justificada': dataset.loc[idx, 'conclusão justificada'],
            'cid': dataset.loc[idx, 'cid'],
            'princípio ativo': dataset.loc[idx, 'princípio ativo'],
            'nome comercial': dataset.loc[idx, 'nome comercial'],
            'tipo da tecnologia': dataset.loc[idx, 'tipo da tecnologia'],
            'órgão': dataset.loc[idx, 'órgão'],
            'serventia': dataset.loc[idx, 'serventia'],
            'referência': dataset.loc[idx, 'link visualização'],
            'similaridade': float(score)
        }
        resultados.append(resultado)
    
    return resultados

#  Consultas de Validação
consultas = [
    "Qual o entendimento sobre o uso de Canabidiol em Goiás?",
    "Qual o entendimento para a liberação de UTI para pacientes com câncer?",
    "Quais são os pareceres sobre procedimentos cirúrgicos em Goiás?",
    "O que foi decidido sobre medicamentos para diabetes tipo 1?",
    "Quais são os principais medicamentos liberados para doenças raras?"
]

#  Execução dos Testes
erros = []
sucessos = []

for consulta in consultas:
    print(f"\n🔎 Consulta: {consulta}")
    resultados = buscar_jurisprudencia(consulta)
    
    if not resultados:
        erros.append((consulta, "Nenhum resultado encontrado."))
        print(" Nenhum resultado encontrado.")
        continue
    
    for resultado in resultados:
        if resultado['similaridade'] < 0.5:
            erros.append((consulta, "Similaridade baixa."))
        elif not all([resultado['diagnóstico'], resultado['conclusão'], resultado['referência']]):
            erros.append((consulta, "Campos faltantes."))
        else:
            sucessos.append((consulta, resultado['diagnóstico'], resultado['referência']))
            print(f" Diagnóstico: {resultado['diagnóstico']}")
            print(f"   Referência: {resultado['referência']}")
            print(f"   Similaridade: {resultado['similaridade']}")

#  Relatório Final
print("\n===== RELATÓRIO FINAL =====")
print(f" Consultas bem-sucedidas: {len(sucessos)}")
print(f" Consultas com erro: {len(erros)}")

if erros:
    print("\n Lista de Erros:")
    for erro in erros:
        print(f"- {erro[0]}: {erro[1]}")
