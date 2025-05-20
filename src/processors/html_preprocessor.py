#!/usr/bin/env python3
"""
Module pour le prétraitement et l'extraction de contenu depuis le HTML.
"""

import re
from bs4 import BeautifulSoup
# Import correct des classes de commentaires et autres types spéciaux
from bs4 import Comment, Declaration, ProcessingInstruction

def preprocess_html(html_content):
    """
    Prétraite le HTML brut en supprimant les éléments inutiles et en extrayant
    uniquement le texte pertinent.
    """
    if not html_content:
        return ""
    
    # Création de l'objet BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Suppression des balises script, style et autres éléments non textuels
    for element in soup(['script', 'style', 'head', 'title', 'meta', 'iframe', 'noscript', 
                        'header', 'footer', 'nav', 'aside']):
        element.extract()
    
    # Suppression des commentaires HTML - Correction de l'erreur ici
    for comment in soup.find_all(text=lambda text: isinstance(text, (
            Comment, Declaration, ProcessingInstruction))):
        comment.extract()
    
    # Extraction du texte brut
    text = soup.get_text(separator=' ')
    
    # Normalisation du texte
    # 1. Remplacement des sauts de ligne multiples par un espace
    text = re.sub(r'\n+', ' ', text)
    
    # 2. Remplacement des tabulations et espaces multiples par un espace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 3. Suppression des espaces au début et à la fin
    text = text.strip()
    
    # 4. Suppression des caractères spéciaux superflus
    text = re.sub(r'[^\w\s.,;:!?«»\'"()[\]{}€$£¥%&@#*=+\-–—/\\]', '', text)
    
    # 5. Normalisation des espaces autour de la ponctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # 6. Suppression des lignes vides
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text

def extract_main_content(html_content):
    """
    Extrait le contenu principal d'une page web en ignorant les menus, sidebars, etc.
    Utilise des heuristiques pour identifier le contenu principal.
    """
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Liste des sélecteurs CSS où le contenu principal se trouve souvent
    main_content_selectors = [
        "article", "main", ".content", "#content", ".post", 
        ".article", ".entry-content", ".post-content", "#main-content"
    ]
    
    main_content = None
    
    # Chercher le contenu principal selon les sélecteurs
    for selector in main_content_selectors:
        content = soup.select(selector)
        if content:
            # Prend le plus long contenu trouvé
            main_content = max(content, key=lambda x: len(x.get_text()))
            break
    
    # Si aucun contenu n'a été trouvé avec les sélecteurs, prendre le <body>
    if not main_content:
        main_content = soup.body
    
    # Si on a trouvé du contenu, le nettoyer
    if main_content:
        # Suppression des éléments non pertinents du contenu principal
        for element in main_content.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            element.extract()
        
        return preprocess_html(str(main_content))
    
    # Si tout échoue, revenir au prétraitement standard
    return preprocess_html(html_content)

def get_page_title(html_content):
    """
    Extrait le titre d'une page HTML.
    """
    if not html_content:
        return "Pas de titre trouvé"
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.title.string if soup.title else "Pas de titre trouvé"
    except:
        return "Pas de titre trouvé"
