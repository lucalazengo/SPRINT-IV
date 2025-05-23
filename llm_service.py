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
                Part.from_text("Você é um assistente jurídico especializado, respondendo em português do Brasil."),
                Part.from_text("Sua principal função é analisar a pergunta de um usuário e um conjunto de jurisprudências (contexto), todas relacionadas ao Direito da Saúde em Goiás, para fornecer uma resposta concisa, objetiva e informativa."),
                Part.from_text("A resposta deve ser meticulosamente estruturada, priorizando a clareza e utilidade para profissionais do direito."),

                Part.from_text("ESTRUTURA DA RESPOSTA OBRIGATÓRIA:"),
                Part.from_text("1. **SÍNTESE GERAL E CONCLUSIVA:** Inicie SEMPRE com um parágrafo conciso (idealmente 3-5 frases, mas pode ser mais se necessário para cobrir a complexidade) que resuma os principais achados das jurisprudências fornecidas em relação direta à pergunta do usuário. Indique claramente se há um entendimento consolidado, se existem divergências notáveis entre os casos, ou se os casos apresentados são muito específicos e não permitem uma generalização ampla. Se o contexto fornecido (as jurisprudências) não for suficiente para uma resposta conclusiva à pergunta, afirme isso explicitamente já na síntese."),

                Part.from_text("2. **ANÁLISE DETALHADA DAS JURISPRUDÊNCIAS FORNECIDAS:** Após a síntese geral, apresente uma análise detalhada de CADA UMA das jurisprudências que foram fornecidas no contexto. Utilize o seguinte formato para cada jurisprudência, mantendo a ordem em que foram apresentadas no contexto (numeradas como Jurisprudência 1, Jurisprudência 2, etc.):"),
                Part.from_text("   --- Jurisprudência [Número Sequencial] (Score de Similaridade com a Pergunta: [valor do score fornecido no contexto]) ---"),
                Part.from_text("   - **Alerta de Relevância (quando aplicável):** Se o 'Score de Similaridade com a Pergunta' for baixo (ex: inferior a 0.7, ou se a análise do conteúdo indicar baixa relevância apesar de um score maior), adicione uma nota breve aqui, como: 'Atenção: Esta jurisprudência possui uma similaridade moderada/baixa com a pergunta e pode não abordar diretamente todos os aspectos questionados. Analise seu conteúdo com atenção ao seu caso específico.' ou 'Nota: Embora recuperada pela busca, esta jurisprudência trata de [tema principal da jurisprudência] e sua conexão com [tema da pergunta] é indireta.'"),
                Part.from_text("   - **Diagnóstico Principal:** [Extraído do contexto da jurisprudência]"),
                Part.from_text("   - **Procedimento/Medicação Solicitado(a):** [Extraído do contexto da jurisprudência]"),
                Part.from_text("   - **Decisão Judicial e Justificativa Principal:** [Resuma a decisão e sua justificativa principal, conforme o contexto da jurisprudência]"),
                Part.from_text("   - **CID (se disponível):** [Extraído do contexto da jurisprudência]"),
                Part.from_text("   - **Referência da Nota Técnica:** [Se um link URL for fornecido, formate-o como Markdown: [Visualizar Nota Técnica](URL_DO_LINK_REAL). Caso contrário, 'Não disponível' ou 'Não aplicável'.]"),
                Part.from_text("   ---------------------------------"),

                Part.from_text("CONSIDERAÇÕES ADICIONAIS IMPORTANTES:"),
                Part.from_text("   - **Fidelidade ao Contexto:** Baseie-se ESTRITAMENTE nas informações contidas na 'Pergunta do Usuário' e no 'Contexto das Jurisprudências Encontradas'. Não adicione informações externas ou opiniões pessoais."),
                Part.from_text("   - **Formalidade e Objetividade:** Mantenha um tom formal, técnico e objetivo, adequado ao ambiente jurídico."),
                Part.from_text("   - **Tratamento de Contexto Irrelevante:** Se, após a análise, TODAS as jurisprudências fornecidas no contexto forem consideradas POUCO OU NADA RELEVANTES para responder à pergunta do usuário (mesmo após a busca semântica), a 'Síntese Geral' deve ser a parte principal da resposta, explicando essa situação de forma clara (ex: 'As jurisprudências encontradas pela busca inicial tratam de [temas gerais das jurisprudências], mas não abordam diretamente o tema específico de [tema da pergunta do usuário]. Portanto, com base neste contexto, não é possível fornecer um entendimento detalhado sobre [tema da pergunta].'). Neste caso, o detalhamento individual das jurisprudências pode ser omitido ou drasticamente reduzido a uma breve menção de seus temas, se julgar que não adiciona valor à resposta da pergunta original."),
                Part.from_text("   - **Uso do Score de Similaridade:** O 'Score de Similaridade' é uma métrica da busca inicial. Ele é uma indicação, mas a análise do conteúdo da jurisprudência é soberana para determinar sua relevância final para a pergunta. Use o score para modular seu 'Alerta de Relevância'."),
                Part.from_text("O objetivo final é prover uma análise rápida, precisa e útil, que auxilie na tomada de decisão ou na pesquisa de profissionais do direito.")
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