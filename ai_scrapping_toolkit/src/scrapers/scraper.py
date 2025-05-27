# ...existing code...

from .anti_scraping import enhanced_selenium_fetch

def fetch_content(url: str, method: str = "auto", wait_time: int = 5,
                 respect_robots: bool = True, user_agent: Optional[str] = None,
                 rate_limit: float = 1.0, random_delay: bool = False,
                 retry_attempts: int = 3) -> Optional[str]:
    """
    Récupère le contenu HTML d'un site web selon la méthode spécifiée.
    
    Args:
        url: URL du site à scraper
        method: Méthode à utiliser ('requests', 'selenium', 'auto')
        wait_time: Temps d'attente après chargement pour Selenium (secondes)
        respect_robots: Si True, vérifie et respecte les règles robots.txt
        user_agent: User-Agent à utiliser pour les requêtes
        rate_limit: Délai minimum entre les requêtes en secondes
        random_delay: Ajouter des délais aléatoires pour simuler un comportement humain
        retry_attempts: Nombre de tentatives en cas d'échec
        
    Returns:
        str or None: Contenu HTML du site ou None en cas d'échec
    """
    # ...existing code for robots.txt checking...
    
    # Choix automatique de la méthode si non spécifiée
    if method == "auto":
        method = _determine_best_method(url)
        logger.info(f"Méthode auto-sélectionnée: {method}")
    
    # Exécution de la méthode choisie
    if method == "selenium":
        return enhanced_selenium_fetch(
            url, wait_time, user_agent, random_delay, retry_attempts
        )
    else:  # method == "requests"
        return _fetch_with_requests(url, user_agent, retry_attempts)

def _fetch_with_requests(url: str, user_agent: Optional[str] = None, 
                        retry_attempts: int = 3) -> Optional[str]:
    """
    Version améliorée du fetch avec requests incluant les tentatives multiples.
    """
    for attempt in range(retry_attempts):
        try:
            headers = {
                'User-Agent': user_agent or 'AI-Scrapping-Toolkit/1.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'fr,fr-FR;q=0.9,en;q=0.8,en-US;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none'
            }
            
            # Délai aléatoire entre les tentatives
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(delay)
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Détection de l'encodage si nécessaire
            if response.encoding == 'ISO-8859-1':
                encoding = response.apparent_encoding
                response.encoding = encoding
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Tentative {attempt + 1}/{retry_attempts} échouée: {e}")
            if attempt == retry_attempts - 1:
                logger.error(f"Toutes les tentatives ont échoué pour {url}")
                return None
    
    return None

# ...existing code...