#!/usr/bin/env python3
"""
Module contenant les différentes méthodes de scraping web.
"""

import time
import requests
from urllib.parse import urlparse

def fetch_with_requests(url, headers=None, timeout=30):
    """
    Récupère le contenu HTML d'une URL en utilisant la bibliothèque requests.
    Adapté pour les sites statiques.
    """
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération avec requests: {e}")
        return None

def fetch_with_selenium(url, wait_time=5, headless=True):
    """
    Récupère le contenu HTML d'une URL en utilisant Selenium avec ChromeDriver.
    Adapté pour les sites dynamiques utilisant JavaScript.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        # Configuration des options Chrome
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Installation et configuration du driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Récupération de la page
        driver.get(url)
        
        # Attente pour le chargement du JavaScript
        time.sleep(wait_time)
        
        # Récupération du contenu
        html_content = driver.page_source
        
        # Fermeture du navigateur
        driver.quit()
        
        return html_content
    except Exception as e:
        print(f"Erreur lors de la récupération avec Selenium: {e}")
        return None

def validate_url(url):
    """
    Valide et normalise une URL.
    """
    # Ajout du protocole si absent
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Vérification de la validité de l'URL
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            print(f"URL invalide: {url}")
            return None
        return url
    except Exception:
        print(f"URL invalide: {url}")
        return None

def fetch_content(url, method="auto", wait_time=5):
    """
    Récupère le contenu HTML d'un site web en utilisant la méthode spécifiée.
    """
    valid_url = validate_url(url)
    if not valid_url:
        return None
    
    html_content = None
    
    if method.lower() == "requests":
        html_content = fetch_with_requests(valid_url)
    elif method.lower() == "selenium":
        html_content = fetch_with_selenium(valid_url, wait_time)
    else:  # mode auto
        html_content = fetch_with_requests(valid_url)
        
        # Si requests échoue ou retourne un contenu qui semble incomplet,
        # on essaie avec Selenium
        if not html_content or "<body" not in html_content.lower():
            print("La récupération avec requests a échoué ou retourné un contenu incomplet.")
            print("Tentative avec Selenium...")
            html_content = fetch_with_selenium(valid_url, wait_time)
    
    return html_content
