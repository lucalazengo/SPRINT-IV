import os
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"c:/Users/mlzengo/Documents/TJGO/SPRINT IV/br-tjgo-cld-02-09b1b22e65b3.json"

#  Inicializando Vertex AI
vertexai.init(project="br-tjgo-cld-02", location="us-central1")

class ConsultaJurisprudencia:
    
    def __init__(self):
        #  Instruções para o modelo (Prompt Reformulado)
        self.system_instruction = [
            "Sempre responda em português do Brasil.",
            "Utilize as informações fornecidas no contexto para responder com clareza e objetividade.",
            "As jurisprudências fornecidas são exclusivamente sobre temas relacionados a Direito à Saúde no Estado de Goiás.",
            "Os principais temas abordados são: Uso de Medicamentos, Internações Hospitalares, UTI, Procedimentos Cirúrgicos, "
            "Tratamento de Doenças Raras e Direito à Vida.",
            "Ao responder, seja específico em citar o diagnóstico, o tipo de procedimento e a decisão tomada na jurisprudência.",
            "Caso existam decisões contraditórias sobre o tema, informe isso claramente.",
            "Formate as respostas da seguinte forma:",
            "1️⃣ Diagnóstico:",
            "2️⃣ Procedimento ou Medicação:",
            "3️⃣ Decisão Judicial:",
            "4️⃣ CID (se disponível):",
            "5️⃣ Referência para a nota técnica (se disponível):",
            "Se não encontrar informações suficientes, responda: 'Não há jurisprudências suficientes sobre este tema.'"
        ]
        
        #  Modelo Gemini inicializado
        self.model = GenerativeModel('gemini-1.5-pro-002', system_instruction=self.system_instruction)
        
        #  Configuração de segurança
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        #  Configuração de geração
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.8,
            "top_p": 0.95,
        }
    
    def consultar(self, texto):
        """
        Realiza uma consulta semântica ao modelo Gemini.
        """
        print(" Iniciando consulta no modelo Gemini...")

        #  Cria um Part para enviar o texto
        pdf_partials = [Part.from_text(texto)]

        #  Tratamento de erro
        try:
            resposta = self.model.generate_content(
                pdf_partials,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return resposta.text
        except Exception as e:
            print(" Erro durante a consulta ao Gemini:")
            print(e)
            return f"Erro ao consultar o modelo: {e}"
