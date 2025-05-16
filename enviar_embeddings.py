import pickle
from consulta_jurisprudencia import ConsultaJurisprudencia

# ✅ Inicializar a classe de consulta
consulta = ConsultaJurisprudencia()

# ✅ Carregar os embeddings e referências
with open('embeddings.pkl', 'rb') as f:
    embeddings, referencias = pickle.load(f)

# ✅ Enviar para o modelo
for index, (embedding, link) in enumerate(zip(embeddings, referencias)):
    print(f"🔄 Enviando embedding {index + 1} de {len(embeddings)}... [Link: {link}]")
    try:
        # Enviar o vetor como um chunk para o modelo Gemini
        resposta = consulta.consultar(f"Referência: {link}\n\nEmbedding: {embedding}")
        print(f"✅ Embedding {index + 1} enviado com sucesso!")
    except Exception as e:
        print(f"❌ Falha ao enviar embedding {index + 1}: {e}")
