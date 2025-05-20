"""
Fonctions pour l'extraction de données à partir de chunks via des modèles de langage.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

def build_extraction_prompt(query: str, data_types: Optional[List[str]] = None) -> str:
    """
    Construit un prompt d'extraction basé sur la requête de l'utilisateur.
    
    Args:
        query (str): Requête utilisateur (ex: "Extraire les titres et dates")
        data_types (List[str], optional): Types de données à extraire explicitement
        
    Returns:
        str: Prompt formaté pour l'extraction de données
    """
    # Extraction des types de données de la requête si non spécifiés
    if not data_types:
        common_data_types = [
            "titre", "date", "auteur", "prix", "description", "url", 
            "image", "catégorie", "tags", "contenu"
        ]
        
        query_lower = query.lower()
        detected_types = []
        
        for data_type in common_data_types:
            if data_type in query_lower or f"{data_type}s" in query_lower:
                detected_types.append(data_type)
        
        data_types = detected_types if detected_types else ["titre", "date", "contenu"]
    
    # Construction du prompt
    data_types_str = ", ".join(data_types)
    
    prompt = (
        f"{query}\n\n"
        f"Extrait les informations suivantes du contenu HTML : {data_types_str}. "
        f"Réponds au format JSON structuré comme ceci : {{\"{data_types[0]}s\": [...], "
    )
    
    if len(data_types) > 1:
        prompt += f"\"{data_types[1]}s\": [...], "
    
    prompt += "...}."
    
    if "titre" in data_types or "titres" in data_types:
        prompt += " Pour les titres, extrait tous les titres significatifs (h1, h2, h3 ou texte en gras qui ressemble à un titre)."
    
    if "date" in data_types or "dates" in data_types:
        prompt += " Pour les dates, extrait toutes les dates au format JJ/MM/AAAA si possible."
    
    if "prix" in data_types or "tarif" in data_types:
        prompt += " Pour les prix, inclus la devise et assure-toi qu'ils sont correctement formatés."
    
    prompt += " Si un élément est introuvable, renvoie une liste vide pour cette catégorie."
    
    return prompt

def extract_data_from_chunks(
    chunks: List[str],
    query: str,
    llm_provider: Any,
    max_workers: int = 4,
    data_types: Optional[List[str]] = None,
    output_format: str = "json"
) -> List[Dict[str, Any]]:
    """
    Extrait des données de chunks HTML via un modèle de langage.
    
    Args:
        chunks (List[str]): Liste de chunks HTML
        query (str): Requête d'extraction en langage naturel
        llm_provider: Provider du modèle de langage
        max_workers (int): Nombre maximum de workers parallèles
        data_types (List[str], optional): Types de données à extraire
        output_format (str): Format de sortie (json, markdown, text)
        
    Returns:
        List[Dict[str, Any]]: Résultats d'extraction pour chaque chunk
    """
    if not chunks:
        logger.warning("Aucun chunk fourni pour l'extraction.")
        return []
    
    # Construction du prompt d'extraction
    extraction_prompt = build_extraction_prompt(query, data_types)
    logger.info(f"Prompt d'extraction: {extraction_prompt}")
    
    # Fonction pour traiter un chunk
    def process_chunk(chunk, idx):
        logger.info(f"Traitement du chunk {idx+1}/{len(chunks)}")
        try:
            result = llm_provider.extract(chunk, extraction_prompt, output_format)
            return {"chunk_id": idx, "result": result, "chunk_text": chunk[:100] + "..."}
        except Exception as e:
            logger.error(f"Erreur d'extraction pour le chunk {idx}: {str(e)}")
            return {"chunk_id": idx, "error": str(e), "chunk_text": chunk[:100] + "..."}
    
    # Extraction parallèle sur tous les chunks
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre les tâches
        futures = {executor.submit(process_chunk, chunk, i): i for i, chunk in enumerate(chunks)}
        
        # Collecter les résultats au fur et à mesure qu'ils se terminent
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Chunk {chunk_idx+1} traité avec succès")
            except Exception as e:
                logger.error(f"Erreur avec le chunk {chunk_idx+1}: {str(e)}")
                results.append({"chunk_id": chunk_idx, "error": str(e)})
    
    # Trier les résultats par chunk_id pour préserver l'ordre original
    results.sort(key=lambda x: x.get("chunk_id", 0))
    
    return results

def aggregate_extraction_results(results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Agrège les résultats d'extraction de plusieurs chunks.
    
    Args:
        results (List[Dict]): Liste des résultats d'extraction par chunk
        
    Returns:
        Dict[str, List[Any]]: Données agrégées par type
    """
    if not results:
        return {}
    
    # Initialiser le dictionnaire agrégé
    aggregated_data = {}
    seen_items = {}  # Pour déduplication
    
    for result_item in results:
        # Ignorer les résultats en erreur
        if "error" in result_item and "result" not in result_item:
            continue
        
        result = result_item.get("result", {})
        
        # Cas où result est une chaîne non parsée
        if isinstance(result, str):
            try:
                # Essayer de parser en JSON
                result = json.loads(result)
            except json.JSONDecodeError:
                # Ignorer les résultats non parsables
                continue
        
        # Cas où la réponse brute est fournie
        if isinstance(result, dict) and "raw_response" in result:
            try:
                # Essayer d'extraire et parser le JSON de la réponse brute
                raw_text = result["raw_response"]
                if "```json" in raw_text:
                    json_text = raw_text.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_text)
                else:
                    # Essai simple de parse du texte brut
                    try:
                        result = json.loads(raw_text)
                    except:
                        continue
            except:
                continue
        
        # Agréger chaque type de données
        for data_type, items in result.items():
            # Ignorer les métadonnées et champs non-liste
            if not isinstance(items, list):
                continue
            
            # Initialiser la liste si elle n'existe pas encore
            if data_type not in aggregated_data:
                aggregated_data[data_type] = []
            
            # Ajouter chaque élément avec déduplication
            for item in items:
                # Créer une clé de hachage pour la déduplication
                if isinstance(item, dict):
                    # Pour les objets, utiliser une représentation JSON triée
                    item_key = json.dumps(item, sort_keys=True)
                else:
                    # Pour les valeurs simples, utiliser directement la valeur
                    item_key = str(item).strip().lower()
                
                # Ajouter l'élément s'il n'a pas déjà été vu
                if item_key not in seen_items:
                    seen_items[item_key] = True
                    aggregated_data[data_type].append(item)
    
    return aggregated_data
