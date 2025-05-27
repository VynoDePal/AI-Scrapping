from .enhanced_data_extractor import enhanced_extract_data_from_chunks

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
        # Utiliser le nouvel extracteur amélioré
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