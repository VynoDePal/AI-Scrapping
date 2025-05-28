"""
Module pour vérifier et respecter les règles robots.txt des sites web.
"""

import urllib.robotparser
import urllib.parse
import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class RobotsChecker:
    """Gère la vérification des règles robots.txt et le rate limiting."""
    
    def __init__(self, user_agent: str = "*", respect_robots: bool = True, 
                 rate_limit: float = 1.0):
        """
        Initialise le vérificateur de robots.txt.
        
        Args:
            user_agent: Identifiant user-agent à utiliser (défaut: "*")
            respect_robots: Si True, respecte les règles robots.txt
            rate_limit: Délai minimum entre les requêtes en secondes
        """
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.rate_limit = rate_limit
        self.parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self.last_request_time: Dict[str, float] = {}
    
    def can_fetch(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si l'URL peut être explorée selon les règles robots.txt.
        
        Args:
            url: URL à vérifier
            
        Returns:
            Tuple[bool, Optional[str]]: (autorisé, raison du refus)
        """
        if not self.respect_robots:
            return True, None
        
        try:
            # Extraire le domaine
            parsed_url = urllib.parse.urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Obtenir ou créer le parser pour ce domaine
            if base_url not in self.parsers:
                logger.info(f"Récupération et analyse du fichier robots.txt de {base_url}")
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(f"{base_url}/robots.txt")
                parser.read()
                self.parsers[base_url] = parser
            
            # Vérifier si l'accès est autorisé
            parser = self.parsers[base_url]
            if not parser.can_fetch(self.user_agent, url):
                logger.warning(f"L'accès à {url} est interdit par robots.txt")
                return False, "Interdit par robots.txt"
            
            # Vérifier le délai entre requêtes (Crawl-delay)
            crawl_delay = parser.crawl_delay(self.user_agent)
            if crawl_delay is not None:
                delay = max(crawl_delay, self.rate_limit)
            else:
                delay = self.rate_limit
            
            # Respecter le délai entre requêtes
            if base_url in self.last_request_time:
                time_since_last = time.time() - self.last_request_time[base_url]
                if time_since_last < delay:
                    wait_time = delay - time_since_last
                    logger.info(f"Attente de {wait_time:.2f}s pour respecter le rate limiting")
                    time.sleep(wait_time)
            
            # Mettre à jour le timestamp de dernière requête
            self.last_request_time[base_url] = time.time()
            return True, None
            
        except Exception as e:
            logger.warning(f"Erreur lors de la vérification de robots.txt pour {url}: {e}")
            # En cas d'erreur, on autorise par défaut mais avec un délai de sécurité
            time.sleep(self.rate_limit)
            return True, f"Erreur de vérification: {str(e)}"
