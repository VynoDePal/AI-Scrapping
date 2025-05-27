"""
Module pour découper le contenu HTML en chunks selon différentes méthodes.
"""

import re
import logging
from bs4 import BeautifulSoup
from typing import List, Optional
from .semantic_chunker import semantic_html_to_chunks

logger = logging.getLogger(__name__)

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
    
    # Méthodes existantes
    if method == "tags":
        return _chunk_by_tags(html_content, max_length)
    elif method == "length":
        return _chunk_by_length(html_content, max_length, overlap)
    else:  # hybrid
        return _chunk_hybrid(html_content, max_length, overlap)

def _chunk_by_tags(html_content: str, max_length: int) -> List[str]:
    """Découpe par balises HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []
    
    for element in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = element.get_text(strip=True)
        if text and len(text) > 50:  # Ignorer les éléments trop courts
            if len(text) <= max_length:
                chunks.append(text)
            else:
                # Diviser les éléments trop longs
                sub_chunks = _split_text(text, max_length)
                chunks.extend(sub_chunks)
    
    return chunks

def _chunk_by_length(html_content: str, max_length: int, overlap: int) -> List[str]:
    """Découpe par longueur fixe."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    return _split_text_with_overlap(text, max_length, overlap)

def _chunk_hybrid(html_content: str, max_length: int, overlap: int) -> List[str]:
    """Découpe hybride combinant structure HTML et longueur."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Identifier les sections importantes
    sections = []
    for element in soup.find_all(['article', 'section', 'div'], class_=re.compile(r'content|main|article')):
        text = element.get_text(separator=' ', strip=True)
        if len(text) > 100:
            sections.append(text)
    
    # Si aucune section identifiée, utiliser le texte complet
    if not sections:
        text = soup.get_text(separator=' ', strip=True)
        sections = [text]
    
    # Découper chaque section
    chunks = []
    for section in sections:
        if len(section) <= max_length:
            chunks.append(section)
        else:
            sub_chunks = _split_text_with_overlap(section, max_length, overlap)
            chunks.extend(sub_chunks)
    
    return chunks

def _split_text(text: str, max_length: int) -> List[str]:
    """Divise un texte en chunks sans chevauchement."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 pour l'espace
        
        if current_length + word_length > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def _split_text_with_overlap(text: str, max_length: int, overlap: int) -> List[str]:
    """Divise un texte en chunks avec chevauchement."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Chercher un point de coupure naturel
        cut_point = text.rfind(' ', start, end)
        if cut_point == -1 or cut_point <= start:
            cut_point = end
        
        chunks.append(text[start:cut_point])
        start = cut_point - overlap
        
        if start < 0:
            start = 0
    
    return chunks
