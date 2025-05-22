import os
import pickle
import re
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify

# Importações do Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold

# --- Configurações da Aplicação ---
app = Flask(__name__)

# --- Constantes e Configurações de Caminhos ---
MODEL_NAME_SEMANTIC = 'all-mpnet-base-v2' # Para a busca semântica
MODEL_NAME_LLM = 'gemini-1.5-pro-002' # Para o LLM Gemini (ajuste conforme necessário)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATASET_FILENAME = 'embeddings.csv'
EMBEDDINGS_FILENAME = 'embeddings.pkl'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)
EMBEDDINGS_PATH = os.path.join(DATA_DIR, EMBEDDINGS_FILENAME)

# --- Configurações do Vertex AI ---
# !! IMPORTANTE !!
# Para DESENVOLVIMENTO, você pode definir o caminho do JSON aqui, mas NUNCA o versione.
# Em PRODUÇÃO, configure a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS no seu servidor
# ou use a autenticação implícita do ambiente Google Cloud.
GOOGLE_CREDENTIALS_PATH = r"c:/Users/mlzengo/Documents/TJGO/SPRINT IV/br-tjgo-cld-02-09b1b22e65b3.json" 
GOOGLE_PROJECT_ID = "br-tjgo-cld-02"  
GOOGLE_LOCATION = "us-central1"

sentence_model = None
dataset = None
embeddings_global = None
faiss_index = None
enriquecedor_llm = None
resources_loaded = False
expected_model_dim = 0
dataset_len = 0

class EnriquecedorLLM:
    def __init__(self, project_id, location):
        try:
            if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and os.path.exists(GOOGLE_CREDENTIALS_PATH):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
            
            vertexai.init(project=project_id, location=location)
            print(f"✅ Vertex AI inicializado para projeto {project_id} em {location}.")
            
            # Instruções de sistema refinadas para o contexto do tribunal
            self.system_instruction = [
                Part.from_text("Você é um assistente jurídico altamente especializado, respondendo em português do Brasil."),
                Part.from_text("Sua principal função é analisar a pergunta de um usuário e um conjunto de jurisprudências (contexto) relacionadas à Direito da Saúde no Estado de Goiás para fornecer uma resposta concisa, objetiva e informativa."),
                Part.from_text("A resposta deve ser estruturada da seguinte forma:"),
                Part.from_text("1. **Síntese Geral:** Comece com um parágrafo resumindo os principais achados das jurisprudências em relação à pergunta do usuário. Destaque se há um entendimento consolidado, divergências, ou se os casos são muito específicos."),
                Part.from_text("2. **Detalhes das Jurisprudências Relevantes (se aplicável e solicitado implicitamente ou se o resumo se beneficiar disso):** Após a síntese, se houver jurisprudências particularmente ilustrativas ou se a pergunta demandar detalhes, apresente até 2-3 casos mais relevantes. Para cada caso detalhado, use o seguinte formato:"),
                Part.from_text("   --- Jurisprudência Detalhada ---"),
                Part.from_text("   - **Diagnóstico Principal:** [extraído do contexto]"),
                Part.from_text("   - **Procedimento/Medicação Solicitado(a):** [extraído do contexto]"),
                Part.from_text("   - **Decisão Judicial e Justificativa Principal:** [resuma a decisão e sua justificativa principal, conforme o contexto]"),
                Part.from_text("   - **CID (se disponível):** [extraído do contexto]"),
                Part.from_text("   - **Score de Similaridade com a Pergunta:** [fornecido no contexto]"),
                Part.from_text("   - **Referência da Nota Técnica:** [Se um link URL for fornecido no contexto para este campo, formate-o como um link clicável em Markdown, por exemplo: [Visualizar Nota Técnica](URL_REAL_DO_LINK). Se não houver link ou não for aplicável, indique 'Não disponível' ou 'Não se aplica'.]"),
                Part.from_text("   ---------------------------------"),
                Part.from_text("Se as jurisprudências fornecidas no contexto não forem suficientes para responder à pergunta do usuário de forma conclusiva, ou se o contexto estiver vazio, declare isso claramente (ex: 'Com base nas informações fornecidas, não foi possível encontrar um entendimento claro ou jurisprudências suficientemente detalhadas sobre este tema específico.' ou 'Nenhuma jurisprudência foi encontrada para esta consulta.')."),
                Part.from_text("Mantenha um tom formal e objetivo. Evite opiniões pessoais ou informações não contidas no contexto fornecido (pergunta do usuário e jurisprudências)."),
                Part.from_text("Use o 'Score de Similaridade' para entender a relevância de cada jurisprudência para a pergunta original, mas não necessariamente o mencione na resposta final, a menos que seja para justificar a escolha de detalhar um caso."),
                Part.from_text("O objetivo é prover uma análise útil e rápida para profissionais do direito dentro do tribunal.")
            ]
            
            self.model = GenerativeModel(
                MODEL_NAME_LLM,
                system_instruction=self.system_instruction
            )
            print(f"✅ Modelo LLM ({MODEL_NAME_LLM}) inicializado.")
            self.model_ready = True

        except Exception as e:
            print(f"❌ Erro ao inicializar EnriquecedorLLM: {e}")
            # import traceback; traceback.print_exc() # Para debug
            self.model_ready = False
            self.model = None # Garante que o modelo não seja usado

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.generation_config = {
            "max_output_tokens": 4096, 
            "temperature": 0.6, 
            "top_p": 0.95,
        }
    
    def gerar_resposta_enriquecida(self, contexto_completo):
        if not self.model_ready or not self.model:
            return "Desculpe, o assistente de enriquecimento de respostas não está disponível no momento."
        
        print("🔄 Gerando resposta enriquecida com LLM...")
        
        try:
            
           
            response = self.model.generate_content(
                [contexto_completo], 
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=False, 
            )
            
            # Tratamento de resposta e possíveis bloqueios
            if response.candidates and response.candidates[0].content.parts:
                generated_text = response.candidates[0].content.parts[0].text
                print("✅ Resposta do LLM recebida.")
                return generated_text
            else:
                reason = "desconhecida"
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason.name
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                print(f"⚠️ Resposta da LLM vazia ou bloqueada. Razão: {reason}")
                if reason == "SAFETY":
                    return "A resposta não pôde ser gerada devido a restrições de segurança do conteúdo."
                return "Não foi possível gerar uma resposta neste momento. Por favor, tente reformular sua pergunta ou tente mais tarde."

        except Exception as e:
            print(f"❌ Erro durante a consulta ao LLM: {e}")
            # import traceback; traceback.print_exc()
            return f"Erro ao consultar o modelo de linguagem: {type(e).__name__}"

# --- Função Auxiliar de Parsing (sem alterações) ---
def extract_field_from_text(text_block, field_label):
    if not isinstance(text_block, str) or not isinstance(field_label, str): return "N/A"
    pattern = rf"(?i){re.escape(field_label)}\s*:\s*(.*?)(?=\n\s*[A-ZÀ-Úa-zÀ-ÖØ-öø-ÿ][\w\sÀ-ÖØ-öø-ÿ()]*\s*:|\Z)"
    match = re.search(pattern, text_block, re.DOTALL)
    if match:
        value = match.group(1).strip()
        if value.startswith(".."): value = value[2:].strip()
        elif value.startswith("."): value = value[1:].strip()
        return value if value else "N/A"
    return "N/A"

# --- 
def load_resources():
    global sentence_model, dataset, embeddings_global, faiss_index, enriquecedor_llm
    global resources_loaded, expected_model_dim, dataset_len

    resources_loaded = False # Reset
    try:
        print("🚀 Inicializando Enriquecedor LLM...")
        enriquecedor_llm = EnriquecedorLLM(project_id="br-tjgo-cld-02", location="us-central1")
        if not enriquecedor_llm.model_ready:
            print("⚠️ Enriquecedor LLM não foi inicializado corretamente. Funcionalidade de enriquecimento estará desabilitada.")
           

        
        print(f"🚀 Carregando modelo SentenceTransformer ({MODEL_NAME_SEMANTIC})...")
        sentence_model = SentenceTransformer(MODEL_NAME_SEMANTIC)
        expected_model_dim = sentence_model.get_sentence_embedding_dimension()
        print(f"✅ Modelo SentenceTransformer carregado. Dimensão: {expected_model_dim}")

        
        print(f"🚀 Carregando DataFrame de {DATASET_PATH}...")
        if not os.path.exists(DATASET_PATH):
            print(f"❌ Erro: Arquivo do dataset não encontrado: {DATASET_PATH}"); return
        dataset = pd.read_csv(DATASET_PATH)
        dataset_len = len(dataset)
        print(f"✅ DataFrame carregado com {dataset_len} registros.")

        # 4. Carregar e processar Embeddings
        print(f"🚀 Carregando embeddings de {EMBEDDINGS_PATH}...")
        if not os.path.exists(EMBEDDINGS_PATH):
            print(f"❌ Erro: Arquivo de embeddings não encontrado: {EMBEDDINGS_PATH}"); return
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings_data_from_pickle = pickle.load(f)
        print("✅ Embeddings carregados do pickle.")
        
        actual_embeddings_source = None
        if isinstance(embeddings_data_from_pickle, tuple):
            print(f"Dados do pickle são uma TUPLA com {len(embeddings_data_from_pickle)} elemento(s).")
            if embeddings_data_from_pickle and isinstance(embeddings_data_from_pickle[0], (list, np.ndarray)):
                actual_embeddings_source = embeddings_data_from_pickle[0]
                print(f"    🎉 Usando o elemento 0 da tupla como fonte de embeddings.")
            else: print("❌ Elemento 0 da tupla não é lista/array ou tupla está vazia."); return
        elif isinstance(embeddings_data_from_pickle, (list, np.ndarray)):
            actual_embeddings_source = embeddings_data_from_pickle
        else: print(f"❌ Tipo de dados inesperado ({type(embeddings_data_from_pickle)}) no pickle."); return

        final_embeddings_array = None
        if isinstance(actual_embeddings_source, np.ndarray):
            if actual_embeddings_source.ndim == 2 and actual_embeddings_source.shape[0] == dataset_len and actual_embeddings_source.shape[1] == expected_model_dim:
                final_embeddings_array = actual_embeddings_source.astype('float32')
        elif isinstance(actual_embeddings_source, list):
            if len(actual_embeddings_source) == dataset_len:
                temp_valid = [list(e) for e in actual_embeddings_source if isinstance(e, (list, np.ndarray)) and len(e) == expected_model_dim]
                if len(temp_valid) == dataset_len: final_embeddings_array = np.array(temp_valid, dtype='float32')
        
        if final_embeddings_array is None or final_embeddings_array.size == 0 :
            print("❌ Falha ao processar embeddings para NumPy array ou dimensões incorretas."); return
        embeddings_global = final_embeddings_array
        print(f"✅ Embeddings processados para array NumPy com shape {embeddings_global.shape}")
        
        faiss.normalize_L2(embeddings_global)
        print("✅ Embeddings normalizados.")
        faiss_index = faiss.IndexFlatIP(embeddings_global.shape[1])
        faiss_index.add(embeddings_global)
        print(f"✅ Índice FAISS criado com {faiss_index.ntotal} vetores.")
        
        resources_loaded = True # Indica que os recursos principais para busca estão OK
        if not (enriquecedor_llm and enriquecedor_llm.model_ready):
            print("⚠️ Aviso: O LLM para enriquecimento de respostas não está operacional.")
            # A aplicação funcionará, mas sem o enriquecimento do LLM.

    except Exception as e:
        import traceback
        print(f"❌ Erro crítico geral ao carregar recursos: {e}")
        traceback.print_exc()
        resources_loaded = False



# Dentro da função buscar_jurisprudencia_semantica em similarity.py

def buscar_jurisprudencia_semantica(query, top_k=5):
    # Alteração aqui: troque 'not dataset' por 'dataset is None'
    if sentence_model is None or faiss_index is None or dataset is None:
        print("⚠️ Aviso: Recursos da busca semântica (sentence_model, faiss_index ou dataset) não carregados.")
        return [] # Retorna lista vazia

    print(f"🔎 Iniciando busca semântica por: '{query}' com top_k={top_k}")
    query_embedding = sentence_model.encode([query], normalize_embeddings=True)
    query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

    distances, indices = faiss_index.search(query_embedding_np, top_k)
    resultados_extraidos = []
    
    
    field_labels_for_parsing = {
        'diagnóstico': 'Diagnóstico', 'conclusão': 'Conclusão',
        'justificativa': 'Justificativa', 
        'cid': 'CID', 'princípio ativo': 'Princípio Ativo', 
        'nome comercial': 'Nome Comercial', 'descrição': 'Descrição', 
        'tipo da tecnologia': 'Tipo da Tecnologia', 'órgão': 'Órgão', 
        'serventia': 'Serventia'
    }

    for i in range(indices.shape[1]): 
        idx = indices[0, i]
        score = distances[0, i]
        if idx < 0 or idx >= len(dataset): continue
        try:
            row = dataset.iloc[idx]
            texto_completo = str(row['texto']) if pd.notna(row['texto']) else ""
            item = {'texto_original': texto_completo} 
            for key, label_in_text in field_labels_for_parsing.items():
                item[key] = extract_field_from_text(texto_completo, label_in_text)
            
            item['referencia'] = str(row['referencia']) if pd.notna(row['referencia']) else ''
            item['similaridade_busca'] = float(score) if pd.notna(score) else 0.0
            resultados_extraidos.append(item)
        except Exception as e:
            print(f"❌ Erro ao processar item da busca semântica no índice {idx}: {e}")
    
    print(f"✅ Busca semântica encontrou {len(resultados_extraidos)} resultados.")
    return resultados_extraidos


# --- Rotas Flask ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_chat_response():
    if not resources_loaded: # Checagem geral de recursos
        return jsonify({'error': 'Serviço temporariamente indisponível (recursos não carregados).'}), 503

    user_query = request.form.get('query')
    if not user_query:
        return jsonify({'error': 'Nenhuma pergunta fornecida.'}), 400

    try:
        # 1. Obter Resultados da Busca Semântica Primeiro
        resultados_semanticos = buscar_jurisprudencia_semantica(user_query, top_k=5)

        # 2. Construir o Contexto para o LLM
        contexto_para_llm = f"Pergunta do Usuário: \"{user_query}\"\n\n"
        if not resultados_semanticos:
            contexto_para_llm += "Contexto das Jurisprudências Encontradas: Nenhuma jurisprudência específica foi encontrada pela busca inicial para esta pergunta."
        else:
            contexto_para_llm += "Contexto das Jurisprudências Encontradas (analise e use para formular sua resposta):\n"
            for i, res in enumerate(resultados_semanticos):
                contexto_para_llm += f"\n--- Jurisprudência {i+1} (Score de Similaridade com a pergunta: {res.get('similaridade_busca', 0):.3f}) ---\n"
                # Usar os rótulos que você listou, pegando os valores do dicionário 'res'
                contexto_para_llm += f"Diagnóstico: {res.get('diagnóstico', 'Não informado')}\n"
                contexto_para_llm += f"Conclusão: {res.get('conclusão', 'Não informado')}\n"
                contexto_para_llm += f"Justificativa: {res.get('justificativa', 'Não informado')}\n" 
                contexto_para_llm += f"CID: {res.get('cid', 'Não informado')}\n"
                contexto_para_llm += f"Princípio Ativo: {res.get('princípio ativo', 'Não informado')}\n"
                contexto_para_llm += f"Nome Comercial: {res.get('nome comercial', 'Não informado')}\n"
                contexto_para_llm += f"Descrição: {res.get('descrição', 'Não informado')}\n"
                contexto_para_llm += f"Tipo da Tecnologia: {res.get('tipo da tecnologia', 'Não informado')}\n"
                contexto_para_llm += f"Órgão: {res.get('órgão', 'Não informado')}\n"
                contexto_para_llm += f"Serventia: {res.get('serventia', 'Não informado')}\n"
                if res.get('referencia'):
                    contexto_para_llm += f"Referência da Nota Técnica: {res.get('referencia')}\n"
                # Opcional: incluir trecho do texto original se for curto e relevante
                # texto_orig_curto = res.get('texto_original', '')[:300] # Primeiros 300 chars
                # contexto_para_llm += f"Trecho do Texto Original: {texto_orig_curto}...\n"

        print(f"\n--- CONTEXTO COMPLETO PARA LLM ---\n{contexto_para_llm}\n--------------------------------\n")

        # 3. Chamar o LLM para gerar uma resposta enriquecida
        if enriquecedor_llm and enriquecedor_llm.model_ready:
            resposta_final = enriquecedor_llm.gerar_resposta_enriquecida(contexto_para_llm)
        else:
            resposta_final = "O assistente de enriquecimento de respostas não está disponível. Seguem os resultados da busca simples:\n\n"
            if resultados_semanticos:
                for i, res in enumerate(resultados_semanticos):
                    resposta_final += f"Resultado {i+1}:\n"
                    for key, value in res.items():
                        if key != 'texto_original' and value != "N/A" and value: # Não mostrar texto original completo aqui
                             resposta_final += f"  {key.replace('_', ' ').capitalize()}: {value}\n"
                    resposta_final += "\n"
            else:
                resposta_final = "Nenhuma jurisprudência encontrada para esta consulta."
        
        # A resposta_final é uma string (do LLM ou do fallback)
        return jsonify({'response': resposta_final, 'type': 'llm_response'}) # Adiciona type para o frontend

    except Exception as e:
        print(f"❌ Erro geral na rota /get_response: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ocorreu um erro crítico ao processar sua solicitação: {str(e)}'}), 500

# --- Execução Principal ---
if __name__ == '__main__':
    load_resources() 
    if not resources_loaded:
        print("‼️ ATENÇÃO: Alguns recursos essenciais podem não ter sido carregados. A aplicação pode não funcionar como esperado.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)