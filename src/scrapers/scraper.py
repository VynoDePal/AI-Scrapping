"""
Module pour récupérer le contenu HTML de sites web.
Fournit des fonctions pour différentes méthodes de scraping.
"""

import logging
import requests
from typing import Optional
import time
import random
import re
from urllib.parse import urlparse

# Configuration du logger
logger = logging.getLogger(__name__)

def fetch_content(url: str, method: str = "auto", wait_time: int = 5,
                 respect_robots: bool = True, user_agent: Optional[str] = None,
                 rate_limit: float = 1.0) -> Optional[str]:
    """
    Récupère le contenu HTML d'un site web selon la méthode spécifiée.
    
    Args:
        url: URL du site à scraper
        method: Méthode à utiliser ('requests', 'selenium', 'auto')
        wait_time: Temps d'attente après chargement pour Selenium (secondes)
        respect_robots: Si True, vérifie et respecte les règles robots.txt
        user_agent: User-Agent à utiliser pour les requêtes
        rate_limit: Délai minimum entre les requêtes en secondes
        
    Returns:
        str or None: Contenu HTML du site ou None en cas d'échec
    """
    # Définir un User-Agent par défaut s'il n'est pas spécifié
    if not user_agent:
        user_agent = "AI-Scrapping-Toolkit/1.0 (+https://github.com/kevyn-odjo/ai-scrapping)"
    
    # Vérifier la conformité avec robots.txt si demandé
    if respect_robots:
        try:
            from .robots_checker import RobotsChecker
            checker = RobotsChecker(user_agent=user_agent, rate_limit=rate_limit)
            can_fetch, reason = checker.can_fetch(url)
            if not can_fetch:
                logger.error(f"Accès interdit à {url}: {reason}")
                return None
        except ImportError:
            logger.warning("Module robots_checker non disponible, vérification robots.txt ignorée")
            # Ajouter un délai simple pour respecter le rate limiting de base
            time.sleep(rate_limit)
    else:
        # Même sans respect de robots.txt, on respecte un délai minimal
        time.sleep(rate_limit)
    
    # Choix automatique de la méthode si non spécifiée
    if method == "auto":
        method = _determine_best_method(url)
        logger.info(f"Méthode auto-sélectionnée: {method}")
    
    # Exécution de la méthode choisie
    if method == "selenium":
        return _fetch_with_selenium(url, wait_time, user_agent)
    else:  # method == "requests"
        return _fetch_with_requests(url, user_agent)

def _determine_best_method(url: str) -> str:
    """
    Détermine la meilleure méthode de scraping en fonction de l'URL.
    Certains sites nécessitent JavaScript (Selenium), d'autres non (Requests).
    
    Args:
        url: URL à analyser
        
    Returns:
        str: Méthode recommandée ('requests', 'selenium')
    """
    # Liste de domaines connus pour nécessiter JavaScript
    js_heavy_domains = ['twitter.com', 'facebook.com', 'instagram.com', 
                       'linkedin.com', 'airbnb.com', 'booking.com']
    
    # Extraction du domaine de l'URL
    domain = urlparse(url).netloc
    
    # Vérification si le domaine est dans la liste qui nécessite JavaScript
    for js_domain in js_heavy_domains:
        if js_domain in domain:
            return "selenium"
    
    # Par défaut, on utilise requests qui est plus rapide
    return "requests"

def _fetch_with_requests(url: str, user_agent: Optional[str] = None) -> Optional[str]:
    """
    Récupère le contenu HTML d'un site web avec la bibliothèque requests.
    
    Args:
        url: URL du site à scraper
        user_agent: User-Agent à utiliser pour les requêtes
        
    Returns:
        str or None: Contenu HTML du site ou None en cas d'échec
    """
    try:
        headers = {
            'User-Agent': user_agent or 'AI-Scrapping-Toolkit/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'fr,fr-FR;q=0.9,en;q=0.8,en-US;q=0.7',
            'Referer': 'https://www.google.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Lève une exception si statut HTTP d'erreur
        
        # Détection de l'encodage si nécessaire
        if response.encoding == 'ISO-8859-1':
            # Tentative de détection plus précise de l'encodage
            encoding = response.apparent_encoding
            response.encoding = encoding
        
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la récupération avec requests: {e}")
        return None

def _fetch_with_selenium(url: str, wait_time: int = 5, user_agent: Optional[str] = None) -> Optional[str]:
    """
    Récupère le contenu HTML d'un site web avec Selenium.
    
    Args:
        url: URL du site à scraper
        wait_time: Temps d'attente après chargement (secondes)
        user_agent: User-Agent à utiliser pour les requêtes
        
    Returns:
        str or None: Contenu HTML du site ou None en cas d'échec
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        import time
        
        # Configuration de Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Définir le user-agent si spécifié
        if user_agent:
            chrome_options.add_argument(f"user-agent={user_agent}")
        
        # Initialisation du driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Définir un timeout
        driver.set_page_load_timeout(30)
        
        # Accéder à l'URL
        driver.get(url)
        
        # Attendre le chargement complet
        time.sleep(wait_time)
        
        # Récupérer le contenu HTML
        html_content = driver.page_source
        
        # Fermer le driver
        driver.quit()
        
        return html_content
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération avec Selenium: {e}")
        return None