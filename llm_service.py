# llm_service.py
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from config import GOOGLE_CREDENTIALS_PATH, MODEL_NAME_LLM 

class EnriquecedorLLM:
    def __init__(self, project_id, location):
        self.model_ready = False
        self.model = None
        try:
            if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and \
               GOOGLE_CREDENTIALS_PATH and os.path.exists(GOOGLE_CREDENTIALS_PATH):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
            
            vertexai.init(project=project_id, location=location)
            print(f" LLM Service: Vertex AI inicializado para projeto {project_id} em {location}.")
            
            self.system_instruction = [
                Part.from_text("Você é um assistente jurídico altamente especializado, respondendo em português do Brasil, para uso interno em um tribunal de justiça."),
                Part.from_text("Sua principal função é analisar a pergunta de um usuário e um conjunto de jurisprudências (contexto) relacionadas à Direito da Saúde no Estado de Goiás para fornecer uma resposta concisa, objetiva e informativa."),
                Part.from_text("A resposta deve ser estruturada da seguinte forma:"),
                Part.from_text("1. **Síntese Geral:** Comece com um parágrafo resumindo os principais achados das jurisprudências em relação à pergunta do usuário. Destaque se há um entendimento consolidado, divergências, ou se os casos são muito específicos."),
                Part.from_text("2. **Detalhes das Jurisprudências Relevantes (se aplicável e a síntese se beneficiar disso):** Após a síntese, se houver jurisprudências particularmente ilustrativas ou se a pergunta demandar detalhes, apresente até 2-3 casos mais relevantes (mesmo que mais jurisprudências tenham sido fornecidas no contexto). Para cada caso detalhado, use o seguinte formato:"),
                Part.from_text("   --- Jurisprudência Detalhada ---"),
                Part.from_text("   - **Diagnóstico Principal:** [extraído do contexto]"),
                Part.from_text("   - **Procedimento/Medicação Solicitado(a):** [extraído do contexto]"),
                Part.from_text("   - **Decisão Judicial e Justificativa Principal:** [resuma a decisão e sua justificativa principal, conforme o contexto]"),
                Part.from_text("   - **CID (se disponível):** [extraído do contexto]"),
                Part.from_text("   - **Score de Similaridade com a Pergunta:** [fornecido no contexto]"),
                Part.from_text("   - **Referência da Nota Técnica:** [Se um link URL for fornecido no contexto para este campo, formate-o como um link clicável em Markdown, por exemplo: [Visualizar Nota Técnica](URL_REAL_DO_LINK). Se não houver link ou não for aplicável, indique 'Não disponível' ou 'Não se aplica'.]"),
                Part.from_text("   ---------------------------------"),
                Part.from_text("Se as jurisprudências fornecidas no contexto não forem suficientes para responder à pergunta do usuário de forma conclusiva, ou se o contexto estiver vazio (Nenhuma jurisprudência específica foi encontrada), declare isso claramente (ex: 'Com base nas informações fornecidas, não foi possível encontrar um entendimento claro ou jurisprudências suficientemente detalhadas sobre este tema específico.' ou 'Nenhuma jurisprudência foi encontrada para esta consulta. Prossiga com cautela ou refaça a pergunta de forma mais ampla.')."),
                Part.from_text("Mantenha um tom formal e objetivo. Evite opiniões pessoais ou informações não contidas no contexto fornecido (pergunta do usuário e jurisprudências)."),
                Part.from_text("Use o 'Score de Similaridade' fornecido no contexto para entender a relevância de cada jurisprudência para a pergunta original ao formular sua síntese e ao decidir quais casos detalhar."),
                Part.from_text("O objetivo é prover uma análise útil e rápida para profissionais do direito dentro do tribunal.")
            ]
            
            self.model = GenerativeModel(
                MODEL_NAME_LLM, 
                system_instruction=self.system_instruction
            )
            print(f" LLM Service: Modelo LLM ({MODEL_NAME_LLM}) inicializado.")
            self.model_ready = True

        except Exception as e:
            print(f" LLM Service: Erro ao inicializar EnriquecedorLLM: {e}")

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
        
        print(" LLM Service: Gerando resposta enriquecida...")
        
        try:
            response = self.model.generate_content(
                [contexto_completo],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=False,
            )
            
            if response.candidates and response.candidates[0].content.parts:
                generated_text = response.candidates[0].content.parts[0].text
                print(" LLM Service: Resposta do LLM recebida.")
                return generated_text
            else:
                reason = "desconhecida"
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason.name
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                print(f" LLM Service: Resposta da LLM vazia ou bloqueada. Razão: {reason}")
                if reason == "SAFETY":
                    return "A resposta não pôde ser gerada devido a restrições de segurança do conteúdo."
                return "Não foi possível gerar uma resposta neste momento. Tente reformular sua pergunta."
        except Exception as e:
            print(f" LLM Service: Erro durante a consulta ao LLM: {e}")
            return f"Erro ao consultar o modelo de linguagem: {type(e).__name__}"