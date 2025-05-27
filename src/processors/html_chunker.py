#!/usr/bin/env python3
"""
Module pour découper le HTML en segments (chunks) de taille gérable 
pour les modèles de langage avec contraintes de tokens.
"""

import re
import logging
from bs4 import BeautifulSoup, Tag
from typing import List
from .semantic_chunker import semantic_html_to_chunks

# Configuration du logger
logger = logging.getLogger(__name__)

def chunk_by_tags(soup, tag_selectors=None, max_length=1000):
    """
    Découpe le contenu HTML par balises spécifiques.
    """
    if tag_selectors is None:
        tag_selectors = ['p', 'div', 'section', 'article', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
    chunks = []
    current_chunk = ""
    heading_context = ""  # Contexte des titres pour chaque section
    
    # Fonction pour ajouter un chunk à la liste
    def add_chunk_to_list(chunk_text):
        if chunk_text.strip():
            # Ajouter le contexte des titres si nécessaire
            if heading_context and not chunk_text.startswith(heading_context):
                final_chunk = f"{heading_context}\n{chunk_text}"
            else:
                final_chunk = chunk_text
            chunks.append(final_chunk.strip())
    
    try:
        # Parcourir tous les éléments de premier niveau dans le body
        elements = soup.find_all(tag_selectors)
        
        # Si aucun élément n'est trouvé, essayer de prendre le texte directement
        if not elements:
            logger.warning("Aucune balise de structuration trouvée dans le HTML. Extraction du texte complet.")
            text = soup.get_text(separator='\n')
            if text and len(text) > max_length:
                # Découper le texte en chunks de taille max_length
                return chunk_by_length(text, max_length=max_length, overlap=100)
            elif text:
                return [text.strip()]
            else:
                return []
        
        for element in elements:
            # Gérer les titres pour maintenir le contexte
            if element.name in ['h1', 'h2', 'h3']:
                heading_text = element.get_text().strip()
                if element.name == 'h1':
                    heading_context = heading_text
                elif element.name == 'h2':
                    # Conserver le h1 précédent s'il existe
                    h1_context = heading_context.split('\n')[0] if '\n' not in heading_context else ""
                    heading_context = f"{h1_context}\n{heading_text}" if h1_context else heading_text
                elif element.name == 'h3' and heading_context:
                    heading_context = f"{heading_context}\n{heading_text}"
            
            element_text = element.get_text().strip()
            
            # Si l'élément seul dépasse la taille max, le découper
            if len(element_text) > max_length:
                # D'abord ajouter le chunk en cours s'il existe
                if current_chunk:
                    add_chunk_to_list(current_chunk)
                    current_chunk = ""
                
                # Découper le texte long en morceaux
                words = element_text.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_length:
                        temp_chunk += f" {word}" if temp_chunk else word
                    else:
                        add_chunk_to_list(temp_chunk)
                        temp_chunk = word
                
                if temp_chunk:
                    add_chunk_to_list(temp_chunk)
            
            # Sinon, l'ajouter au chunk en cours ou créer un nouveau chunk
            elif len(current_chunk) + len(element_text) + 1 <= max_length:
                current_chunk += f"\n{element_text}" if current_chunk else element_text
            else:
                add_chunk_to_list(current_chunk)
                current_chunk = element_text
        
        # Ajouter le dernier chunk s'il reste du texte
        if current_chunk:
            add_chunk_to_list(current_chunk)
            
    except Exception as e:
        logger.error(f"Erreur lors du découpage par balises: {e}")
        # Essayer une méthode alternative en cas d'erreur
        text = soup.get_text(separator='\n')
        if text:
            logger.info("Tentative de découpage par longueur suite à une erreur")
            return chunk_by_length(text, max_length=max_length, overlap=100)
    
    # Si aucun chunk n'a été généré, essayer avec le texte brut
    if not chunks:
        logger.warning("Le découpage par balises n'a produit aucun chunk. Tentative avec le texte brut.")
        text = soup.get_text(separator='\n').strip()
        if text:
            # Si le texte est court, le retourner directement
            if len(text) <= max_length:
                return [text]
            # Sinon, le découper par longueur
            return chunk_by_length(text, max_length=max_length, overlap=100)
    
    return chunks

def chunk_by_length(text, max_length=1000, overlap=100):
    """
    Découpe un texte en segments de taille maximum spécifiée avec chevauchement.
    """
    if not text:
        return []
    
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Déterminer la fin du chunk actuel
        end = start + max_length
        
        if end >= len(text):
            # Dernier chunk
            chunks.append(text[start:])
            break
        
        # Chercher la dernière occurrence d'un caractère de séparation
        # pour éviter de couper au milieu d'un mot ou d'une phrase
        separators = ['. ', '? ', '! ', '\n\n', '\n', '. ', ', ', ' ']
        chunk_end = end
        
        for sep in separators:
            last_sep_pos = text[start:end].rfind(sep)
            if last_sep_pos != -1:
                chunk_end = start + last_sep_pos + len(sep)
                break
        
        # Si aucun séparateur n'est trouvé, découper au dernier espace
        if chunk_end == end:
            last_space = text[start:end].rfind(' ')
            if last_space != -1:
                chunk_end = start + last_space + 1
        
        # Si toujours pas de bon point de coupure, couper à la position max
        if chunk_end == end and end < len(text):
            chunk_end = end
        
        # Ajouter le chunk
        chunks.append(text[start:chunk_end])
        
        # Mettre à jour la position de départ avec chevauchement
        start = chunk_end - overlap if chunk_end > start + overlap else chunk_end
    
    return chunks

def html_to_chunks(
    html_content: str, 
    method: str = "hybrid", 
    max_length: int = 1000, 
    overlap: int = 100
) -> List[str]:
    """
    Convertit le contenu HTML en chunks selon la méthode spécifiée.
    
    Args:
        html_content: Contenu HTML brut
        method: Méthode de chunking ('tags', 'length', 'hybrid', 'semantic')
        max_length: Taille maximale d'un chunk
        overlap: Chevauchement entre chunks adjacents
        
    Returns:
        List[str]: Liste des chunks générés
    """
    if not html_content:
        return []
    
    # Nouvelle méthode sémantique
    if method == "semantic":
        return semantic_html_to_chunks(
            html_content, 
            max_length=max_length, 
            overlap=overlap,
            prioritize_important=True
        )
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Supprimer les éléments non textuels
        for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
            tag.decompose()
        
        if method == 'tags':
            logger.debug("Découpage par balises")
            return chunk_by_tags(soup, max_length=max_length)
        elif method == 'length':
            # Extraire tout le texte et découper par longueur
            logger.debug("Découpage par longueur")
            text = soup.get_text(separator='\n')
            return chunk_by_length(text, max_length=max_length, overlap=overlap)
        else:  # method == 'hybrid'
            # D'abord découper par tags, puis par longueur si nécessaire
            logger.debug("Découpage hybride")
            tag_chunks = chunk_by_tags(soup, max_length=max_length * 2)  # Permet des chunks plus grands
            
            final_chunks = []
            for chunk in tag_chunks:
                if len(chunk) > max_length:
                    # Redécouper les chunks trop grands
                    final_chunks.extend(chunk_by_length(chunk, max_length, overlap))
                else:
                    final_chunks.append(chunk)
            
            # Si aucun chunk n'a été généré, essayer le découpage par longueur
            if not final_chunks:
                logger.warning("Le découpage hybride n'a produit aucun chunk. Tentative avec le texte brut.")
                text = soup.get_text(separator='\n').strip()
                if text:
                    return chunk_by_length(text, max_length=max_length, overlap=overlap)
            
            return final_chunks
    except Exception as e:
        logger.error(f"Erreur lors du chunking HTML: {e}")
        # En cas d'erreur, essayer de retourner le texte complet si suffisamment court
        try:
            clean_text = re.sub(r'<[^>]+>', ' ', html_content)  # Suppression grossière des balises
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalisation des espaces
            
            if len(clean_text) <= max_length:
                logger.info("Retour du texte complet nettoyé après erreur")
                return [clean_text]
            else:
                # Découper en chunks de taille fixe en dernier recours
                logger.info("Découpage du texte nettoyé en chunks de taille fixe")
                chunks = []
                for i in range(0, len(clean_text), max_length - overlap):
                    chunks.append(clean_text[i:i + max_length])
                return chunks
        except:
            logger.error("Impossible de récupérer du texte après erreur de chunking")
            return []
