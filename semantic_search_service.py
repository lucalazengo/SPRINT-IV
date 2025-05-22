# semantic_search_service.py
import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME_SEMANTIC, DATASET_PATH, EMBEDDINGS_PATH 
from utils import extract_field_from_text 

class SemanticSearcher:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.embeddings_global = None
        self.index = None
        self.expected_model_dim = 0
        self.dataset_len = 0
        self.is_ready = False

        try:
            print(f" Semantic Search Service: Carregando modelo SentenceTransformer ({MODEL_NAME_SEMANTIC})...")
            self.model = SentenceTransformer(MODEL_NAME_SEMANTIC)
            self.expected_model_dim = self.model.get_sentence_embedding_dimension()
            print(f" Semantic Search Service: Modelo SentenceTransformer carregado. Dimensão: {self.expected_model_dim}")

            print(f" Semantic Search Service: Carregando DataFrame de {DATASET_PATH}...")
            if not os.path.exists(DATASET_PATH):
                print(f" Semantic Search Service: Arquivo do dataset não encontrado: {DATASET_PATH}"); return
            self.dataset = pd.read_csv(DATASET_PATH)
            self.dataset_len = len(self.dataset)
            print(f" Semantic Search Service: DataFrame carregado com {self.dataset_len} registros.")

            print(f" Semantic Search Service: Carregando embeddings de {EMBEDDINGS_PATH}...")
            if not os.path.exists(EMBEDDINGS_PATH):
                print(f" Semantic Search Service: Arquivo de embeddings não encontrado: {EMBEDDINGS_PATH}"); return
            with open(EMBEDDINGS_PATH, 'rb') as f: embeddings_data_from_pickle = pickle.load(f)
            print(" Semantic Search Service: Embeddings carregados do pickle.")
            
            actual_embeddings_source = None
            if isinstance(embeddings_data_from_pickle, tuple):
                print(f"Dados do pickle são uma TUPLA com {len(embeddings_data_from_pickle)} elemento(s).")
                if embeddings_data_from_pickle and isinstance(embeddings_data_from_pickle[0], (list, np.ndarray)):
                    actual_embeddings_source = embeddings_data_from_pickle[0]
                    print(f"    Usando o elemento 0 da tupla como fonte de embeddings.")
                else: print(" Elemento 0 da tupla não é lista/array ou tupla está vazia."); return
            elif isinstance(embeddings_data_from_pickle, (list, np.ndarray)):
                actual_embeddings_source = embeddings_data_from_pickle
            else: print(f" Tipo de dados inesperado ({type(embeddings_data_from_pickle)}) no pickle."); return

            final_embeddings_array = None
            if isinstance(actual_embeddings_source, np.ndarray):
                if actual_embeddings_source.ndim == 2 and actual_embeddings_source.shape[0] == self.dataset_len and actual_embeddings_source.shape[1] == self.expected_model_dim:
                    final_embeddings_array = actual_embeddings_source.astype('float32')
            elif isinstance(actual_embeddings_source, list):
                if len(actual_embeddings_source) == self.dataset_len:
                    temp_valid = [list(e) for e in actual_embeddings_source if isinstance(e, (list, np.ndarray)) and len(e) == self.expected_model_dim]
                    if len(temp_valid) == self.dataset_len: final_embeddings_array = np.array(temp_valid, dtype='float32')
            
            if final_embeddings_array is None or final_embeddings_array.size == 0 :
                print(" Semantic Search Service: Falha ao processar embeddings para NumPy array ou dimensões incorretas."); return
            self.embeddings_global = final_embeddings_array
            print(f" Semantic Search Service: Embeddings processados para array NumPy com shape {self.embeddings_global.shape}")
            
            faiss.normalize_L2(self.embeddings_global)
            print("Semantic Search Service: Embeddings normalizados.")
            self.index = faiss.IndexFlatIP(self.embeddings_global.shape[1])
            self.index.add(self.embeddings_global)
            print(f" Semantic Search Service: Índice FAISS criado com {self.index.ntotal} vetores.")
            self.is_ready = True

        except Exception as e:
            print(f" Semantic Search Service: Erro ao inicializar: {e}")
            import traceback; traceback.print_exc()
            self.is_ready = False

    def search(self, query, top_k=3):
        if not self.is_ready or self.model is None or self.index is None or self.dataset is None:
            print(" Semantic Search Service: Não está pronto ou recursos não carregados.")
            return []

        print(f" Semantic Search Service: Buscando por: '{query}' com top_k={top_k}")
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

        distances, indices = self.index.search(query_embedding_np, top_k)
        resultados_extraidos = []
        
        field_labels_for_parsing = {
            'diagnóstico': 'Diagnóstico', 'conclusão': 'Conclusão',
            'justificativa': 'Justificativa', 'cid': 'CID', 
            'princípio ativo': 'Princípio Ativo', 'nome comercial': 'Nome Comercial', 
            'descrição': 'Descrição', 'tipo da tecnologia': 'Tipo da Tecnologia', 
            'órgão': 'Órgão', 'serventia': 'Serventia'
        }

        for i_loop in range(indices.shape[1]): 
            idx = indices[0, i_loop]
            score = distances[0, i_loop]
            if idx < 0 or idx >= len(self.dataset): continue
            try:
                row = self.dataset.iloc[idx]
                texto_completo = str(row['texto']) if pd.notna(row['texto']) else ""
                item = {'texto_original': texto_completo} 
                for key, label_in_text in field_labels_for_parsing.items():
                    item[key] = extract_field_from_text(texto_completo, label_in_text)
                
                item['referencia'] = str(row['referencia']) if pd.notna(row['referencia']) else ''
                item['similaridade_busca'] = float(score) if pd.notna(score) else 0.0
                resultados_extraidos.append(item)
            except Exception as e:
                print(f" Semantic Search Service: Erro ao processar item no índice {idx}: {e}")
        
        print(f" Semantic Search Service: Busca encontrou {len(resultados_extraidos)} resultados.")
        return resultados_extraidos