"""
Module pour contourner les mesures anti-scraping de manière éthique et légale.
"""

import time
import random
import logging
from typing import Dict, List, Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger(__name__)

class AntiScrapingHandler:
    """Gestionnaire pour contourner les mesures anti-scraping."""
    
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0"
        ]
    
    def get_random_user_agent(self) -> str:
        """Retourne un User-Agent aléatoire."""
        return random.choice(self.user_agents)
    
    def add_human_behavior(self, driver, min_delay: float = 1.0, max_delay: float = 3.0):
        """Ajoute un comportement humain au navigateur."""
        # Délai aléatoire
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
        
        # Simulation de mouvements de souris (optionnel)
        try:
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(driver)
            
            # Mouvement de souris aléatoire
            x_offset = random.randint(-100, 100)
            y_offset = random.randint(-100, 100)
            actions.move_by_offset(x_offset, y_offset).perform()
            
            # Petit délai supplémentaire
            time.sleep(random.uniform(0.5, 1.5))
            
        except Exception as e:
            logger.debug(f"Impossible de simuler les mouvements de souris: {e}")
    
    def handle_access_denied(self, driver, url: str) -> bool:
        """Tente de gérer les pages d'accès refusé."""
        try:
            page_source = driver.page_source.lower()
            
            # Détecter les messages d'accès refusé
            access_denied_indicators = [
                "access denied", "accès refusé", "403 forbidden", 
                "blocked", "bloqué", "captcha", "cloudflare",
                "please enable javascript", "bot detection"
            ]
            
            if any(indicator in page_source for indicator in access_denied_indicators):
                logger.warning(f"Accès refusé détecté pour {url}")
                
                # Attendre plus longtemps pour laisser passer les vérifications
                self.add_human_behavior(driver, 5.0, 10.0)
                
                # Essayer de rafraîchir la page
                driver.refresh()
                time.sleep(random.uniform(3.0, 6.0))
                
                # Vérifier si le problème persiste
                new_page_source = driver.page_source.lower()
                if not any(indicator in new_page_source for indicator in access_denied_indicators):
                    logger.info("Accès rétabli après rafraîchissement")
                    return True
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la gestion de l'accès refusé: {e}")
            return False
    
    def wait_for_content_load(self, driver, timeout: int = 10) -> bool:
        """Attend que le contenu principal soit chargé."""
        try:
            # Attendre que la page soit complètement chargée
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Attendre que le contenu principal soit présent
            content_selectors = [
                "main", "article", "[role='main']", ".content", 
                "#content", ".main-content", "body"
            ]
            
            for selector in content_selectors:
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.debug(f"Contenu trouvé avec le sélecteur: {selector}")
                    return True
                except TimeoutException:
                    continue
            
            logger.warning("Aucun contenu principal détecté, mais la page semble chargée")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'attente du chargement: {e}")
            return False

def enhanced_selenium_fetch(url: str, wait_time: int = 10, user_agent: Optional[str] = None, 
                          random_delay: bool = True, retry_attempts: int = 3) -> Optional[str]:
    """
    Version améliorée du fetch Selenium avec gestion anti-scraping.
    """
    handler = AntiScrapingHandler()
    
    for attempt in range(retry_attempts):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            
            # Configuration de Chrome améliorée
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # User-Agent aléatoire ou spécifique
            if not user_agent:
                user_agent = handler.get_random_user_agent()
            chrome_options.add_argument(f"user-agent={user_agent}")
            
            # Autres options pour éviter la détection
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")  # Charge plus vite
            
            # Initialiser le driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Masquer les traces de Selenium
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Timeout plus long
            driver.set_page_load_timeout(30)
            
            # Délai aléatoire avant la requête
            if random_delay:
                initial_delay = random.uniform(1.0, 3.0)
                time.sleep(initial_delay)
            
            # Accéder à l'URL
            logger.info(f"Tentative {attempt + 1}/{retry_attempts} pour {url}")
            driver.get(url)
            
            # Ajouter un comportement humain
            handler.add_human_behavior(driver, 2.0, 4.0)
            
            # Vérifier les mesures anti-scraping
            if not handler.handle_access_denied(driver, url):
                logger.error(f"Accès toujours refusé après gestion pour {url}")
                driver.quit()
                if attempt < retry_attempts - 1:
                    # Délai plus long avant la prochaine tentative
                    time.sleep(random.uniform(5.0, 10.0))
                    continue
                return None
            
            # Attendre le chargement du contenu
            if not handler.wait_for_content_load(driver, wait_time):
                logger.warning("Le contenu pourrait ne pas être complètement chargé")
            
            # Attendre le temps spécifié
            time.sleep(wait_time)
            
            # Récupérer le contenu
            html_content = driver.page_source
            driver.quit()
            
            # Vérifier la qualité du contenu
            if len(html_content) < 1000:
                logger.warning(f"Contenu suspicieusement court ({len(html_content)} caractères)")
                if attempt < retry_attempts - 1:
                    time.sleep(random.uniform(3.0, 6.0))
                    continue
            
            logger.info(f"Contenu récupéré avec succès ({len(html_content)} caractères)")
            return html_content
            
        except Exception as e:
            logger.error(f"Erreur lors de la tentative {attempt + 1}: {e}")
            if 'driver' in locals():
                try:
                    driver.quit()
                except:
                    pass
            
            if attempt < retry_attempts - 1:
                # Délai exponentiel entre les tentatives
                delay = (2 ** attempt) + random.uniform(1.0, 3.0)
                logger.info(f"Attente de {delay:.1f}s avant la prochaine tentative")
                time.sleep(delay)
    
    logger.error(f"Échec de toutes les tentatives pour {url}")
    return None
