import os
import time
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Caminho absoluto do diretório do script
base_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(base_dir, "notas_tecnicas_goias.csv")

class ENatjusScraper:
    def __init__(self):
        self.url = "https://www.pje.jus.br/e-natjus/pesquisaPublica.php"
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.resultados = []

    def iniciar_navegacao(self):
        logging.info("Iniciando navegador...")
        self.driver.get(self.url)
        
        
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.ID, "btnPesquisar"))
        )

        # Selecionar o filtro para Goiás (GO)
        logging.info("Selecionando o filtro para Goiás...")
        select_element = self.driver.find_element(By.ID, "txtNatResponsavel")
        options = select_element.find_elements(By.TAG_NAME, "option")
        
        for option in options:
            if option.text == "GO - Goiás":
                option.click()
                break

        # Clicar no botão de pesquisa
        botao_pesquisar = self.driver.find_element(By.ID, "btnPesquisar")
        botao_pesquisar.click()

        # Espera para os resultados carregarem
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#tbody tr"))
        )
        logging.info("Resultados carregados com sucesso!")

    def coletar_dados(self):
        logging.info("Coletando dados das notas técnicas...")

        while True:
            linhas = self.driver.find_elements(By.CSS_SELECTOR, "#tbody tr")
            
            if not linhas:
                logging.warning("Nenhuma linha encontrada. Verifique o seletor ou se a página está carregada corretamente.")
                break
            
            for linha in linhas:
                cols = linha.find_elements(By.TAG_NAME, "td")
                if len(cols) > 5 and "GO" in cols[4].text:
                    nota_tecnica = {
                        "ID": cols[0].text,
                        "Data": cols[1].text,
                        "Tecnologia": cols[2].text,
                        "CID": cols[3].text,
                        "NatJus Responsável": cols[4].text,
                        "Status": cols[5].text,
                        "Link Visualização": cols[6].find_elements(By.TAG_NAME, "a")[0].get_attribute("href"),
                        "Link Download": cols[6].find_elements(By.TAG_NAME, "a")[1].get_attribute("href")
                    }
                    self.resultados.append(nota_tecnica)
            
            logging.info(f"{len(self.resultados)} notas técnicas encontradas até agora...")

            # Verifica se o botão "Próxima Página" está habilitado
            try:
                botao_proxima = self.driver.find_element(By.LINK_TEXT, "Próximo")
                if "disabled" in botao_proxima.get_attribute("class"):
                    logging.info("Não há mais páginas disponíveis.")
                    break
                else:
                    botao_proxima.click()
                    time.sleep(2)  # Pequeno delay para carregar
            except Exception as e:
                logging.warning(f"Erro ao tentar avançar para a próxima página: {e}")
                break

    def salvar_dados(self):
        logging.info("Salvando dados coletados em CSV...")
        df = pd.DataFrame(self.resultados)
        df.to_csv(output_csv, index=False)
        logging.info(f"Dados salvos em '{output_csv}'.")

    def executar(self):
        self.iniciar_navegacao()
        self.coletar_dados()
        self.salvar_dados()
        self.driver.quit()
        logging.info("Processo finalizado com sucesso!")
        
if __name__ == "__main__":
    scraper = ENatjusScraper()
    scraper.executar()
