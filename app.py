from flask import Flask, render_template, request, redirect, url_for
from consulta_jurisprudencia import ConsultaJurisprudencia
import pandas as pd
import pickle
import os
from datetime import datetime

# ✅ Inicialização do Flask e do Modelo
app = Flask(__name__)
consulta_gemini = ConsultaJurisprudencia()

# ✅ Inicialização do Cache
cache_file = 'cache_respostas.pkl'
historico_csv = 'historico_consultas.csv'

# ✅ Verificação do arquivo de cache
if os.path.exists(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            cache_respostas = pickle.load(f)
    except EOFError:
        print("⚠️ Arquivo de cache corrompido. Recriando...")
        cache_respostas = {}
else:
    cache_respostas = {}

# ✅ Função para salvar logs
def salvar_log(pergunta, resposta, tempo_resposta):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        'Data': [timestamp],
        'Pergunta': [pergunta],
        'Resposta': [resposta],
        'Tempo de Resposta (s)': [tempo_resposta]
    }

    # ✅ Salva no CSV de histórico
    if not os.path.exists(historico_csv):
        pd.DataFrame(log_data).to_csv(historico_csv, index=False)
    else:
        pd.DataFrame(log_data).to_csv(historico_csv, mode='a', header=False, index=False)

# 🚀 Rota principal (Chat)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pergunta = request.form['pergunta']

        # ✅ Verifica no Cache
        if pergunta in cache_respostas:
            resposta = cache_respostas[pergunta]
            tempo_resposta = "Cache"
            print("✅ Resposta recuperada do cache.")
        else:
            # ✅ Executa a consulta
            tempo_inicio = datetime.now()
            resposta = consulta_gemini.consultar(pergunta)
            tempo_fim = datetime.now()
            tempo_resposta = (tempo_fim - tempo_inicio).total_seconds()

            # ✅ Adiciona ao cache
            cache_respostas[pergunta] = resposta
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_respostas, f)

            # ✅ Salva no log e auditoria
            salvar_log(pergunta, resposta, tempo_resposta)
        
        return render_template('index.html', pergunta=pergunta, resposta=resposta, historico=cache_respostas)
    return render_template('index.html', historico=cache_respostas)

# 🚀 Rota para o Histórico Completo
@app.route('/historico')
def historico_page():
    if os.path.exists(historico_csv):
        historico = pd.read_csv(historico_csv)
        return render_template('historico.html', historico=historico.to_dict(orient='records'))
    return render_template('historico.html', historico=[])

# 🚀 Executar o app
if __name__ == '__main__':
    app.run(debug=True)
