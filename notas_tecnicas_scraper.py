import os
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (NoSuchElementException, 
                                      TimeoutException, 
                                      StaleElementReferenceException)
from urllib.parse import urlparse, parse_qs
import json
from datetime import datetime

class NotasTecnicasScraper:
    def __init__(self):
        self.setup_directories()
        self.setup_driver()
        self.session = requests.Session()
        self.setup_requests_headers()
        
    def setup_directories(self):
        """Cria diretórios para armazenamento"""
        self.download_dir = os.path.join(os.getcwd(), "notas_tecnicas")
        os.makedirs(self.download_dir, exist_ok=True)
        
    def setup_driver(self):
        """Configura o WebDriver do Selenium"""
        chrome_options = Options()
        
        # Configurações para reduzir detecção como bot
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Configurar User-Agent realista
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
    def setup_requests_headers(self):
        """Configura headers para as requisições de download"""
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3',
            'Connection': 'keep-alive',
            'Referer': 'https://www.pje.jus.br/e-natjus/pesquisaPublica.php',
        })
        
    def random_delay(self, min_sec=1, max_sec=3):
        """Atraso aleatório entre ações para parecer humano"""
        time.sleep(random.uniform(min_sec, max_sec))
        
    def wait_for_element(self, by, value, timeout=10):
        """Espera por elemento com tratamento de exceções"""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
        except TimeoutException:
            print(f"Elemento não encontrado: {value}")
            return None
            
    def navigate_to_page(self):
        """Navega para a página de pesquisa"""
        url = "https://www.pje.jus.br/e-natjus/pesquisaPublica.php"
        self.driver.get(url)
        self.random_delay()
        
        # Aceitar cookies se necessário
        try:
            cookie_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Aceitar')]")
            cookie_btn.click()
            self.random_delay()
        except NoSuchElementException:
            pass
            
    def apply_filters(self):
        """Aplica filtros para notas técnicas de Goiás"""
        # Selecionar "GO - Goiás" no dropdown
        dropdown = self.wait_for_element(By.ID, "txtNatResponsavel")
        if dropdown:
            dropdown.click()
            self.random_delay(0.5, 1)
            
            option = self.wait_for_element(By.XPATH, "//option[contains(text(), 'GO - Goiás')]")
            if option:
                option.click()
                self.random_delay()
                
        # Clicar em pesquisar
        pesquisar_btn = self.wait_for_element(By.ID, "btnPesquisar")
        if pesquisar_btn:
            pesquisar_btn.click()
            self.random_delay(2, 4)  # Espera maior para carregar resultados
            
    def extract_links_from_page(self):
        """Extrai links de download da página atual"""
        links = []
        
        # Esperar tabela carregar
        table = self.wait_for_element(By.ID, "tabela-notaTecnica")
        if not table:
            return links
            
        rows = table.find_elements(By.TAG_NAME, "tr")
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 7:  # Verifica se tem coluna de ações
                    continue
                    
                # Extrair metadados
                id_nota = cells[0].text.strip()
                data = cells[1].text.strip()
                tecnologia = cells[2].text.strip()
                cid = cells[3].text.strip()
                
                # Extrair link de download
                download_btn = cells[6].find_element(By.CSS_SELECTOR, "a.btn-download")
                href = download_btn.get_attribute("href")
                
                links.append({
                    'id': id_nota,
                    'data': data,
                    'tecnologia': tecnologia,
                    'cid': cid,
                    'url': href
                })
            except (NoSuchElementException, StaleElementReferenceException) as e:
                print(f"Erro ao extrair linha: {e}")
                continue
                
        return links
        
    def navigate_pagination(self):
        """Navega pelas páginas de resultados"""
        current_page = 1
        max_pages = 100  # Limite de segurança
        all_links = []
        
        while current_page <= max_pages:
            print(f"Processando página {current_page}...")
            
            # Extrair links da página atual
            page_links = self.extract_links_from_page()
            all_links.extend(page_links)
            
            # Tentar ir para próxima página
            try:
                next_btn = self.driver.find_element(By.XPATH, "//a[contains(text(), 'Próximo') and not(contains(@class, 'disabled'))]")
                if next_btn:
                    next_btn.click()
                    self.random_delay(2, 4)  # Espera para carregar nova página
                    current_page += 1
                else:
                    break
            except NoSuchElementException:
                break
            except Exception as e:
                print(f"Erro na paginação: {e}")
                break
                
        return all_links
        
    def download_pdf(self, url, filepath):
        """Baixa um arquivo PDF"""
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Download concluído: {filepath}")
        except requests.RequestException as e:
            print(f"Erro ao baixar {url}: {e}")
        except Exception as e:
            print(f"Erro inesperado: {e}")
        finally:
            self.random_delay(1, 2)  # Atraso após download
            self.session.close()
            self.setup_requests_headers()  # Reconfigura headers após download
            self.session = requests.Session()  # Reinicia sessão
            self.setup_requests_headers()  # Reconfigura headers após download
            self.random_delay(1, 2)  # Atraso após download
            self.session = requests.Session()  # Reinicia sessão
            self.setup_requests_headers()  # Reconfigura headers após download
            