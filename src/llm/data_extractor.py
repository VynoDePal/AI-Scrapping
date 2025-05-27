"""
Module pour l'extraction de données structurées à partir de chunks avec les LLMs.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .enhanced_data_extractor import enhanced_extract_data_from_chunks

logger = logging.getLogger(__name__)

def extract_data_from_chunks(
    chunks: List[str],
    query: str,
    llm_provider,
    max_workers: int = 4,
    enhanced_mode: bool = True,
    url: str = ""
) -> List[Dict[str, Any]]:
    """
    Extrait des données structurées à partir d'une liste de chunks.
    
    Args:
        chunks: Liste des chunks de texte à analyser
        query: Requête d'extraction en langage naturel
        llm_provider: Instance du provider LLM
        max_workers: Nombre maximum de workers pour le traitement parallèle
        enhanced_mode: Utiliser le mode amélioré avec deux passes
        url: URL source pour la détection du type de site
        
    Returns:
        Liste des résultats d'extraction ou résultat agrégé en mode amélioré
    """
    if enhanced_mode:
        logger.info("Utilisation du mode d'extraction amélioré (deux passes)")
        result = enhanced_extract_data_from_chunks(
            chunks, query, llm_provider, url, max_workers
        )
        return [result]  # Retourner dans une liste pour compatibilité
    
    # Mode classique : traiter tous les chunks
    logger.info(f"Extraction de données depuis {len(chunks)} chunks avec {max_workers} workers")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les chunks pour traitement
        future_to_chunk = {
            executor.submit(_extract_from_single_chunk, chunk, query, llm_provider): i
            for i, chunk in enumerate(chunks)
        }
        
        # Collecter les résultats
        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.debug(f"Chunk {chunk_index} traité avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du traitement du chunk {chunk_index}: {e}")
    
    logger.info(f"Extraction terminée: {len(results)} résultats obtenus sur {len(chunks)} chunks")
    return results

def _extract_from_single_chunk(chunk: str, query: str, llm_provider) -> Optional[Dict[str, Any]]:
    """
    Extrait des données d'un chunk individuel.
    
    Args:
        chunk: Chunk de texte à analyser
        query: Requête d'extraction
        llm_provider: Provider LLM à utiliser
        
    Returns:
        Dictionnaire avec les données extraites ou None en cas d'échec
    """
    try:
        # Créer un prompt structuré pour l'extraction
        extraction_prompt = f"""
Analyse ce contenu et extrait les informations demandées selon cette requête : {query}

Réponds uniquement avec un objet JSON valide, sans texte avant ou après.
Si tu ne trouves pas d'information, renvoie un objet avec des tableaux vides.

Contenu à analyser :
{chunk}
"""
        
        result = llm_provider.extract(chunk, extraction_prompt, "json")
        
        if isinstance(result, dict):
            return result
        else:
            logger.warning(f"Résultat inattendu du LLM: {type(result)}")
            return None
            
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction d'un chunk: {e}")
        return None

def aggregate_extraction_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrège les résultats d'extraction de plusieurs chunks.
    
    Args:
        results: Liste des résultats d'extraction
        
    Returns:
        Dictionnaire avec les résultats agrégés
    """
    if not results:
        return {}
    
    # Si un seul résultat, le retourner directement
    if len(results) == 1:
        return results[0]
    
    aggregated = {}
    
    # Agréger les listes
    for result in results:
        for key, value in result.items():
            if key not in aggregated:
                aggregated[key] = []
            
            if isinstance(value, list):
                aggregated[key].extend(value)
            elif value is not None and value != "":
                aggregated[key].append(value)
    
    # Nettoyer et dédupliquer
    for key, value in aggregated.items():
        if isinstance(value, list):
            # Supprimer les doublons tout en préservant l'ordre
            seen = set()
            unique_items = []
            for item in value:
                if isinstance(item, str):
                    item_lower = item.lower().strip()
                    if item_lower not in seen and item_lower:
                        seen.add(item_lower)
                        unique_items.append(item.strip())
                elif item not in unique_items:
                    unique_items.append(item)
            aggregated[key] = unique_items
    
    return aggregated