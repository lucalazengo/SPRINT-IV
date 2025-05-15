from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 🚀 Configuração do Flask
app = Flask(__name__)

# 🚀 Carregar o modelo e os dados
model = SentenceTransformer('all-mpnet-base-v2')
dataset = pd.read_csv('./data/processed/DATASET_FINAL_EMBEDDINGS.csv')

# 🚀 Carregar os embeddings e criar o índice
with open('./data/processed/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# ✅ Corrigido: Nome alterado para evitar conflito com Flask
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

# ✅ Índice atualizado para evitar conflito de nome
faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

print(f"✅ Índice de busca criado com {faiss_index.ntotal} jurisprudências indexadas.")

# 🔎 Função de busca semântica
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

# 🚀 Rota principal de busca
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        resultados = buscar_jurisprudencia(query)
        return render_template('index.html', resultados=resultados, query=query)
    return render_template('index.html')

# 🚀 Rota do Dashboard
@app.route('/dashboard')
def dashboard():
    total_pareceres = len(dataset)
    distribuicao_conclusao = dataset['conclusão justificada'].value_counts()
    distribuicao_tecnologia = dataset['tipo da tecnologia'].value_counts()
    principais_diagnosticos = dataset['diagnóstico'].value_counts().head(5)
    principais_cidades = dataset['cidade'].value_counts().head(5)
    principais_serventias = dataset['serventia'].value_counts().head(5)

    return render_template('dashboard.html',
                           total_pareceres=total_pareceres,
                           distribuicao_conclusao=distribuicao_conclusao,
                           distribuicao_tecnologia=distribuicao_tecnologia,
                           principais_diagnosticos=principais_diagnosticos,
                           principais_cidades=principais_cidades,
                           principais_serventias=principais_serventias)
                           
# 🚀 Executar o app
if __name__ == '__main__':
    app.run(debug=True)
