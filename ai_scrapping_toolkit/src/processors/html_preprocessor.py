"""
Module pour le prétraitement du contenu HTML.
"""

import re
import logging
from bs4 import BeautifulSoup
from typing import Optional

logger = logging.getLogger(__name__)

def preprocess_html(html_content: str) -> str:
    """
    Prétraite le contenu HTML pour extraire uniquement le texte pertinent.
    
    Args:
        html_content: Contenu HTML brut
        
    Returns:
        str: Texte nettoyé et formaté
    """
    if not html_content:
        return ""
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Supprimer les éléments non pertinents
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Extraire le texte
        text = soup.get_text(separator='\n', strip=True)
        
        # Nettoyer le texte
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normaliser les sauts de ligne
        text = re.sub(r' +', ' ', text)  # Normaliser les espaces
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement HTML: {e}")
        return html_content
