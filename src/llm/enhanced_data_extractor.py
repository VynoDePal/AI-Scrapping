"""
Module d'extraction de données amélioré (version de base).
Version complète à implémenter selon les besoins.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def enhanced_extract_data_from_chunks(
    chunks: List[str],
    query: str,
    llm_provider,
    url: str = "",
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Version de base de l'extraction améliorée.
    
    Args:
        chunks: Liste des chunks de texte à analyser
        query: Requête d'extraction en langage naturel
        llm_provider: Instance du provider LLM
        url: URL source pour la détection du type de site
        max_workers: Nombre maximum de workers
        
    Returns:
        Dictionnaire avec les données extraites et agrégées
    """
    logger.info(f"Extraction améliorée de base avec {len(chunks)} chunks")
    
    # Pour l'instant, utiliser la méthode classique mais traiter plus de chunks
    from .data_extractor import _extract_from_single_chunk
    
    results = []
    
    # Traiter plus de chunks (pas seulement le premier)
    chunks_to_process = chunks[:min(10, len(chunks))]  # Traiter jusqu'à 10 chunks
    
    for i, chunk in enumerate(chunks_to_process):
        try:
            result = _extract_from_single_chunk(chunk, query, llm_provider)
            if result:
                results.append(result)
                logger.debug(f"Chunk {i+1}/{len(chunks_to_process)} traité")
        except Exception as e:
            logger.error(f"Erreur chunk {i}: {e}")
    
    # Agrégation simple
    if not results:
        return {}
    
    if len(results) == 1:
        return results[0]
    
    # Agrégation basique
    aggregated = {}
    for result in results:
        for key, value in result.items():
            if key not in aggregated:
                aggregated[key] = []
            
            if isinstance(value, list):
                aggregated[key].extend(value)
            elif value is not None and value != "":
                if isinstance(aggregated[key], list):
                    aggregated[key].append(value)
                else:
                    aggregated[key] = [aggregated[key], value]
    
    # Nettoyer les doublons
    for key, value in aggregated.items():
        if isinstance(value, list):
            # Supprimer les doublons
            unique_items = []
            for item in value:
                if item not in unique_items:
                    unique_items.append(item)
            aggregated[key] = unique_items
    
    return aggregated
