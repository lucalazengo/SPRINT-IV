import pandas as pd
from consulta_jurisprudencia import ConsultaJurisprudencia

#  Carregar o dataset completo
dataset = pd.read_csv('./data/processed/DATASET_FINAL_EMBEDDINGS.csv')

#  Inicializando o modelo de consulta
consulta = ConsultaJurisprudencia()

#  Função para dividir em partes (chunks) de 2048 tokens
def dividir_em_chunks(texto, max_tokens=2048):
    chunks = []
    while len(texto) > 0:
        chunks.append(texto[:max_tokens])
        texto = texto[max_tokens:]
    return chunks

#  Preparar o conteúdo para envio
conteudo_completo = ""
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
    conteudo_completo += bloco

#  Dividir o conteúdo em chunks e enviar para o modelo
chunks = dividir_em_chunks(conteudo_completo)

#  Envio para o modelo Gemini
for chunk in chunks:
    resposta = consulta.consultar(chunk)
    print(" Chunk enviado com sucesso!")
