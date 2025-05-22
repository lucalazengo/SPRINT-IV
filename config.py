import os

# --- Modelos ---
MODEL_NAME_SEMANTIC = 'all-mpnet-base-v2'
MODEL_NAME_LLM = 'gemini-1.5-pro-002'

BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 


DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATASET_FILENAME = 'embeddings.csv'
EMBEDDINGS_FILENAME = 'embeddings.pkl'

DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)
EMBEDDINGS_PATH = os.path.join(DATA_DIR, EMBEDDINGS_FILENAME)


GOOGLE_CREDENTIALS_PATH = r"c:/Users/mlzengo/Documents/TJGO/SPRINT IV/br-tjgo-cld-02-09b1b22e65b3.json" 
GOOGLE_PROJECT_ID = "br-tjgo-cld-02"
GOOGLE_LOCATION = "us-central1"

TOP_K_SEMANTIC_SEARCH = 5 