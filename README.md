# Assistente de Jurisprudência

## Descrição

Este projeto é uma aplicação web Flask projetada para atuar como um assistente inteligente na busca e análise de jurisprudências relacionadas ao Direito da Saúde no Estado de Goiás. A aplicação utiliza busca semântica para encontrar os documentos mais relevantes para a pergunta do usuário e, em seguida, emprega um Modelo de Linguagem Grande (LLM), especificamente o Gemini do Google via Vertex AI, para analisar o contexto encontrado, fornecer uma síntese e detalhar as jurisprudências de forma estruturada e informativa.

O objetivo principal é fornecer uma ferramenta de apoio rápido e preciso para profissionais do direito dentro do contexto do tribunal.

## Funcionalidades Principais

* **Interface de Chat Interativa:** Permite que os usuários façam perguntas em linguagem natural.
* **Busca Semântica Avançada:** Utiliza `Sentence Transformers` para gerar embeddings de texto e `FAISS` para realizar buscas rápidas por similaridade em um corpus de jurisprudências.
* **Extração de Informações:** Parseia automaticamente campos relevantes (Diagnóstico, Conclusão, CID, etc.) a partir do texto das jurisprudências encontradas.
* **Enriquecimento com LLM (Gemini):**
    * Gera uma síntese geral dos achados com base na pergunta e nas jurisprudências recuperadas.


## Configuração e Instalação

### Pré-requisitos

* Python (versão 3.9 ou superior recomendada)
* `pip` (gerenciador de pacotes Python)
* Docker e Docker Compose (Opcional, para execução em container)
* Uma conta Google Cloud Platform (GCP) com:
    * Um projeto GCP criado.
    * A API Vertex AI habilitada.
    * Um arquivo JSON de credenciais de conta de serviço com permissões para usar a Vertex AI (papel "Usuário da Vertex AI" ou similar).

### Passos de Instalação Local

1.  **Clone o Repositório (se aplicável) ou Crie a Estrutura de Pastas:**
    Certifique-se de que todos os arquivos (`.py`, `data/`, `templates/`) estejam na estrutura correta.

2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instale as Dependências:**
    Certifique-se de ter o arquivo `requirements.txt` na raiz do projeto.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuração do Google Cloud / Vertex AI:**
    * **Arquivo de Credenciais:**
        * Coloque seu arquivo JSON de credenciais do GCP em uma pasta segura, por exemplo, crie uma pasta `secrets/` na raiz do projeto e coloque o arquivo lá (ex: `secrets/minha-chave-gcp.json`).
        * **Importante:** Adicione `secrets/` ao seu arquivo `.gitignore` para nunca versionar suas credenciais.
    * **Atualize `config.py`:**
        * Abra o arquivo `config.py`.
        * Verifique e atualize as constantes `GOOGLE_CREDENTIALS_PATH` para apontar para o caminho correto do seu arquivo JSON de credenciais.
        * Atualize `GOOGLE_PROJECT_ID` com o ID do seu projeto GCP.
        * Verifique se `GOOGLE_LOCATION` (ex: "us-central1") está correta.
    * **Alternativa (Recomendada): Variável de Ambiente:**
        Em vez de definir `GOOGLE_CREDENTIALS_PATH` no código, você pode definir a variável de ambiente `GOOGLE_APPLICATION_CREDENTIALS` no seu sistema para apontar para o caminho do arquivo JSON. O script `llm_service.py` já tenta usar essa variável de ambiente se `GOOGLE_CREDENTIALS_PATH` não for encontrado ou não for definido.
        ```bash
        # Exemplo Linux/macOS
        export GOOGLE_APPLICATION_CREDENTIALS="/caminho/para/seu/secrets/minha-chave-gcp.json"
        # Exemplo Windows (PowerShell)
        $env:GOOGLE_APPLICATION_CREDENTIALS="C:\caminho\para\seu\secrets\minha-chave-gcp.json"
        ```

5.  **Arquivos de Dados:**
    * Certifique-se de que os arquivos `embeddings.csv` e `embeddings.pkl` estão localizados na pasta `data/processed/`.
    * Verifique se `DATASET_FILENAME` e `EMBEDDINGS_FILENAME` em `config.py` correspondem aos nomes dos seus arquivos.

## Executando a Aplicação

### Localmente (Sem Docker)

1.  Certifique-se de que seu ambiente virtual está ativado e todas as dependências estão instaladas.
2.  Navegue até a raiz do projeto no terminal.
3.  Execute o aplicativo Flask:
    ```bash
    python app.py
    ```
4.  Abra seu navegador e acesse `http://127.0.0.1:5000` (ou o endereço IP da sua máquina na rede, como `http://10.10.15.7:5000`, conforme mostrado nos seus logs).

### Com Docker Compose (Recomendado para Ambiente Isolado)

1.  Certifique-se de que Docker e Docker Compose estão instalados.
2.  Crie a pasta `secrets` na raiz do projeto e coloque seu arquivo JSON de credenciais do GCP nela (ex: `secrets/dsadf_12344.json` - o nome deve corresponder ao que está no `docker-compose.yml`).
3.  No terminal, na raiz do projeto, execute:
    ```bash
    docker-compose up --build
    ```
    * O `--build` reconstrói a imagem se for a primeira vez ou se o `Dockerfile` ou arquivos copiados mudaram.
4.  Acesse `http://localhost:5000` no seu navegador.
5.  Para parar: `Ctrl+C` no terminal e depois `docker-compose down`.

## Como Usar

Após iniciar a aplicação, acesse a URL fornecida (geralmente `http://localhost:5000`). Você verá uma interface de chat. Digite sua pergunta sobre jurisprudência de Direito da Saúde em Goiás no campo de entrada e clique em "Enviar" ou pressione Enter. O sistema realizará uma busca semântica, enviará os resultados para o LLM Gemini para análise e enriquecimento, e exibirá a resposta formatada.

## Visão Geral dos Módulos

* **`app.py`**: Orquestra a aplicação Flask. Define as rotas web, carrega os serviços na inicialização e lida com as requisições do usuário, coordenando a busca semântica e a chamada ao LLM.
* **`config.py`**: Centraliza todas as configurações globais, como nomes de modelos, caminhos de arquivos e credenciais do GCP.
* **`utils.py`**: Contém funções utilitárias genéricas, como `extract_field_from_text` para parsear informações de blocos de texto.
* **`semantic_search_service.py`**: Encapsula toda a lógica da busca semântica. É responsável por carregar o modelo SentenceTransformer, o dataset, os embeddings, criar o índice FAISS e realizar as buscas por similaridade, retornando os resultados parseados.
* **`llm_service.py`**: Contém a classe `EnriquecedorLLM`, responsável por inicializar o cliente Vertex AI, definir o prompt de sistema para o modelo Gemini e gerar as respostas enriquecidas com base no contexto fornecido.

## Tecnologias Utilizadas

* **Python**: Linguagem de programação principal.
* **Flask**: Microframework web para o backend e a interface.
* **Sentence Transformers**: Para geração de embeddings de texto.
* **FAISS**: Para buscas eficientes por similaridade em vetores de alta dimensão.
* **Google Vertex AI (Gemini 1.5 Pro)**: Modelo de Linguagem Grande para análise, síntese e geração de respostas enriquecidas.
* **Pandas**: Para manipulação de dados (carregamento do CSV).
* **NumPy**: Para operações numéricas, especialmente com embeddings.
* **Docker & Docker Compose**: Para containerização e gerenciamento do ambiente da aplicação.
* **HTML, CSS, JavaScript**: Para a interface do chat no frontend.
* **Marked.js**: Biblioteca JavaScript para renderizar Markdown no frontend.
