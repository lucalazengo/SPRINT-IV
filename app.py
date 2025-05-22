# app.py
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd 
import config
from semantic_search_service import SemanticSearcher
from llm_service import EnriquecedorLLM

app = Flask(__name__)

search_service = None
llm_service = None
resources_fully_loaded = False

def load_all_resources():
    global search_service, llm_service, resources_fully_loaded
    
    print("--- Iniciando Carregamento de Todos os Recursos ---")
    try:
        search_service = SemanticSearcher()
        if not search_service.is_ready:
            print("‼ ATENÇÃO: Semantic Search Service não pôde ser inicializado.")
           
        else:
            print(" Semantic Search Service carregado com sucesso.")

        llm_service = EnriquecedorLLM(project_id=config.GOOGLE_PROJECT_ID, location=config.GOOGLE_LOCATION)
        if not llm_service.model_ready:
            print(" ATENÇÃO: LLM Service (EnriquecedorLLM) não pôde ser inicializado ou o modelo não está pronto.")
            
        else:
            print(" LLM Service carregado com sucesso.")
        
        if search_service and search_service.is_ready: 
            resources_fully_loaded = True
            print("--- Carregamento de Recursos Essenciais Concluído ---")
        else:
            resources_fully_loaded = False
            print("‼ FALHA NO CARREGAMENTO DE RECURSOS ESSENCIAIS (Busca Semântica). A aplicação pode não funcionar.")

    except Exception as e:
        print(f" Erro catastrófico durante o load_all_resources: {e}")
        import traceback
        traceback.print_exc()
        resources_fully_loaded = False


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_chat_response():
    if not resources_fully_loaded or not search_service or not search_service.is_ready:
        return jsonify({'error': 'Serviço temporariamente indisponível (recursos de busca não carregados).'}), 503

    user_query = request.form.get('query')
    if not user_query:
        return jsonify({'error': 'Nenhuma pergunta fornecida.'}), 400

    try:
        resultados_semanticos = search_service.search(user_query, top_k=config.TOP_K_SEMANTIC_SEARCH)

        contexto_para_llm = f"Pergunta do Usuário: \"{user_query}\"\n\n"
        if not resultados_semanticos:
            contexto_para_llm += "Contexto das Jurisprudências Encontradas: Nenhuma jurisprudência específica foi encontrada pela busca inicial para esta pergunta."
        else:
            contexto_para_llm += "Contexto das Jurisprudências Encontradas (analise e use para formular sua resposta):\n"
            for i, res in enumerate(resultados_semanticos):
                contexto_para_llm += f"\n--- Jurisprudência {i+1} (Score de Similaridade com a pergunta: {res.get('similaridade_busca', 0):.3f}) ---\n"
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
        
        print(f"\n--- CONTEXTO COMPLETO PARA LLM ---\n{contexto_para_llm}\n--------------------------------\n")

        if llm_service and llm_service.model_ready:
            resposta_final = llm_service.gerar_resposta_enriquecida(contexto_para_llm)
        else:
            print(" LLM Service não disponível ou não pronto. Usando fallback.")
            resposta_final = "O assistente LLM não está disponível. Seguem os resultados da busca simples:\n\n"
            if resultados_semanticos:
                for i, res in enumerate(resultados_semanticos):
                    resposta_final += f"**Resultado {i+1} (Similaridade: {res.get('similaridade_busca', 0):.2f})**\n"
                    for key, value in res.items():
                        if key not in ['texto_original', 'similaridade_busca'] and value != "N/A" and value:
                             resposta_final += f"  {key.replace('_', ' ').capitalize()}: {value}\n"
                    if res.get('referencia'):
                        resposta_final += f"  Referência: [{res.get('referencia')}]({res.get('referencia')})\n" 
                    resposta_final += "\n"
            else:
                resposta_final = "Nenhuma jurisprudência encontrada para esta consulta."
        
        return jsonify({'response': resposta_final, 'type': 'llm_response'})

    except Exception as e:
        print(f" Erro geral na rota /get_response: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ocorreu um erro crítico ao processar sua solicitação: {str(e)}'}), 500

if __name__ == '__main__':
    load_all_resources() 
    if not resources_fully_loaded:
        print("‼ ATENÇÃO: APLICAÇÃO INICIADA COM FALHA NO CARREGAMENTO DE RECURSOS ESSENCIAIS.")
    else:
        print(" Aplicação pronta para receber requisições.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)