"""
Module pour extraire le contenu principal et les informations importantes des pages HTML.
"""

import logging
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

def get_page_title(html_content: str) -> Optional[str]:
    """
    Extrait le titre de la page HTML.
    
    Args:
        html_content: Contenu HTML brut
        
    Returns:
        str or None: Titre de la page ou None si non trouvé
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        title_tag = soup.find('title')
        
        if title_tag:
            return title_tag.text.strip()
        
        # Si pas de tag title, essayer les h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.text.strip()
            
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du titre: {e}")
        return None

def extract_main_content(html_content: str) -> Optional[str]:
    """
    Extrait le contenu principal d'une page HTML en éliminant la navigation,
    les publicités, le footer, etc.
    
    Args:
        html_content: Contenu HTML brut
        
    Returns:
        str or None: Contenu principal extrait ou None en cas d'échec
    """
    if not html_content:
        return None
        
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Supprimer les éléments non pertinents
        for element in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style', 'iframe']):
            element.decompose()
        
        # Recherche des conteneurs principaux communs
        main_containers = [
            soup.find('main'),
            soup.find('article'),
            soup.find(id=re.compile(r'content|main|article', re.I)),
            soup.find(class_=re.compile(r'content|main|article|post', re.I)),
            soup.find('div', class_=re.compile(r'content|main|article|post', re.I))
        ]
        
        # Utiliser le premier conteneur trouvé
        for container in main_containers:
            if container:
                # Nettoyer davantage le conteneur
                for element in container.find_all(['script', 'style', 'iframe', 'aside']):
                    element.decompose()
                
                # Vérifier si le contenu est suffisant
                content = container.get_text(separator='\n', strip=True)
                if len(content) > 200:  # Seuil arbitraire pour un article valide
                    return content
        
        # Solution de repli: utiliser l'algorithme heuristique pour trouver la div avec le plus de texte
        main_div = find_main_content_div(soup)
        if main_div:
            content = main_div.get_text(separator='\n', strip=True)
            if len(content) > 200:
                return content
        
        # Si toutes les tentatives échouent, retourner le contenu entier
        logger.warning("Impossible d'identifier le contenu principal spécifique, retour du contenu complet nettoyé")
        
        # Nettoyer le body complet
        body = soup.find('body')
        if body:
            for element in body.find_all(['script', 'style', 'iframe', 'header', 'footer', 'nav']):
                element.decompose()
            return body.get_text(separator='\n', strip=True)
            
        return soup.get_text(separator='\n', strip=True)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du contenu principal: {e}")
        return None

def find_main_content_div(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Trouve la div contenant probablement le contenu principal en se basant sur la densité de texte.
    
    Args:
        soup: Objet BeautifulSoup de la page
        
    Returns:
        BeautifulSoup or None: Élément contenant le contenu principal
    """
    # Obtenir toutes les divs de la page
    all_divs = soup.find_all('div')
    
    if not all_divs:
        return None
    
    # Calculer le score de chaque div (longueur du texte / nombre de balises)
    div_scores = []
    
    for div in all_divs:
        text_length = len(div.get_text(strip=True))
        tags_count = len(div.find_all())
        
        # Éviter la division par zéro
        if tags_count == 0:
            tags_count = 1
            
        # Calculer le score
        content_score = text_length / tags_count
        
        # Bonus pour les divs contenant des paragraphes
        p_count = len(div.find_all('p'))
        if p_count > 3:
            content_score *= 1.5
        
        # Malus pour les divs avec beaucoup de liens
        links_count = len(div.find_all('a'))
        if links_count > 5 and text_length > 0:
            content_score *= (text_length / (text_length + links_count * 50))
        
        div_scores.append((div, content_score, text_length))
    
    # Trier par score et choisir la div avec le score le plus élevé
    sorted_divs = sorted(div_scores, key=lambda x: (x[1], x[2]), reverse=True)
    
    # Retourner la div avec le score le plus élevé si elle a un contenu significatif
    if sorted_divs and sorted_divs[0][2] > 200:
        return sorted_divs[0][0]
    
    return None
