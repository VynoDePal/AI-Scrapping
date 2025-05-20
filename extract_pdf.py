#!/usr/bin/env python3
"""
Script pour extraire des données structurées de fichiers PDF en utilisant des modèles de langage.
"""

import argparse
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional

from src.processors import extract_text_from_pdf, pdf_to_chunks, extract_pdf_metadata
from src.llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_from_pdf(
    pdf_path: str,
    query: str,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    chunk_size: int = 2000,
    max_chunks: Optional[int] = None,
    chunk_method: str = "hybrid",
    api_key: Optional[str] = None,
    output_file: Optional[str] = None,
    temperature: float = 0.0,
    verbose: bool = False,
    extract_metadata: bool = False,
    host: Optional[str] = None,
    timeout: int = 180,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Extrait des données structurées à partir d'un fichier PDF.
    
    Args:
        pdf_path: Chemin du fichier PDF
        query: Requête d'extraction (par exemple "Extraire tous les titres et dates")
        provider: Provider du modèle de langage
        model: Modèle à utiliser
        chunk_size: Taille maximale des chunks
        max_chunks: Nombre maximum de chunks à traiter
        chunk_method: Méthode de chunking (pages, size, hybrid)
        api_key: Clé API pour le provider
        output_file: Fichier de sortie pour les résultats
        temperature: Température pour la génération
        verbose: Afficher des informations détaillées
        extract_metadata: Extraire également les métadonnées du PDF
        host: URL du serveur API (pour lmstudio et ollama)
        timeout: Délai d'attente pour les requêtes LLM en secondes
        max_retries: Nombre maximum de tentatives en cas d'erreur
    
    Returns:
        Dict[str, Any]: Données extraites
    """
    logger.info(f"Traitement du fichier PDF: {pdf_path}")
    
    # Vérifier que le fichier existe et est un PDF
    if not os.path.exists(pdf_path):
        logger.error(f"Le fichier {pdf_path} n'existe pas")
        return {"error": "file_not_found", "message": f"Le fichier {pdf_path} n'existe pas"}
    
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"Le fichier {pdf_path} n'est pas un PDF")
        return {"error": "invalid_format", "message": "Le fichier doit être au format PDF"}
    
    results = {}
    
    # Extraire les métadonnées si demandé
    if extract_metadata:
        logger.info("Extraction des métadonnées du PDF")
        try:
            metadata = extract_pdf_metadata(pdf_path)
            results["metadata"] = metadata
            
            if verbose:
                print("\n=== Métadonnées du PDF ===")
                for key, value in metadata.items():
                    print(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des métadonnées: {e}")
            results["metadata_error"] = str(e)
    
    # Diviser le PDF en chunks
    logger.info(f"Division du PDF en chunks (méthode: {chunk_method}, taille max: {chunk_size})")
    chunks = pdf_to_chunks(pdf_path, method=chunk_method, max_length=chunk_size, overlap=200)
    
    if not chunks:
        logger.error("Échec de l'extraction du texte du PDF ou aucun texte trouvé")
        return {"error": "extraction_failed", "message": "Impossible d'extraire du texte du PDF"}
    
    logger.info(f"{len(chunks)} chunks générés")
    
    # Limiter le nombre de chunks si demandé
    if max_chunks and max_chunks > 0 and max_chunks < len(chunks):
        logger.info(f"Limitation à {max_chunks} chunks sur {len(chunks)}")
        chunks = chunks[:max_chunks]
    
    # Initialiser le provider LLM
    llm_config = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature
    }
    
    # Ajouter des options spécifiques selon le provider
    if host:
        llm_config["host"] = host
    
    if provider == "lmstudio" or provider == "openrouter":
        llm_config["timeout"] = timeout
    
    if provider == "lmstudio":
        llm_config["max_retries"] = max_retries
    
    try:
        llm_provider = get_llm_provider(provider, **llm_config)
    except ImportError as e:
        logger.error(f"Erreur: {str(e)}")
        return {"error": "provider_error", "message": f"Erreur du provider: {str(e)}"}
    
    # Extraire les données des chunks
    logger.info(f"Extraction des données avec {provider} ({model})")
    logger.info(f"Requête: {query}")
    
    extraction_results = extract_data_from_chunks(
        chunks=chunks,
        query=query,
        llm_provider=llm_provider,
        max_workers=min(4, len(chunks))
    )
    
    # Agréger les résultats
    logger.info("Agrégation des résultats")
    extracted_data = aggregate_extraction_results(extraction_results)
    
    # Fusionner avec les métadonnées si elles ont été extraites
    if "metadata" in results:
        extracted_data["pdf_metadata"] = results["metadata"]
    
    # Afficher les statistiques et détails si demandé
    stats = {}
    for data_type, items in extracted_data.items():
        if isinstance(items, list):
            stats[data_type] = len(items)
            
            if verbose and len(items) > 0:
                print(f"\n=== {data_type} ({len(items)} éléments) ===")
                for i, item in enumerate(items[:5]):
                    print(f"  {i+1}. {json.dumps(item, ensure_ascii=False)}")
                if len(items) > 5:
                    print(f"  ... et {len(items) - 5} autres éléments")
    
    logger.info(f"Extraction terminée: {json.dumps(stats)}")
    
    # Sauvegarder les résultats dans un fichier si demandé
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Résultats sauvegardés dans {output_file}")
    
    return extracted_data

def main():
    parser = argparse.ArgumentParser(
        description="Extrait des données structurées à partir d'un fichier PDF en utilisant des modèles de langage"
    )
    parser.add_argument(
        "pdf_path",
        help="Chemin du fichier PDF à analyser"
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Requête d'extraction en langage naturel (ex: 'Extraire tous les titres et dates')"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama", "huggingface", "lmstudio", "openrouter"],
        default="openai",
        help="Provider du modèle de langage"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Modèle à utiliser (ex: gpt-3.5-turbo, llama2, mistral)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Taille maximale des chunks"
    )
    parser.add_argument(
        "--chunk-method",
        choices=["pages", "size", "hybrid"],
        default="hybrid",
        help="Méthode de découpage en chunks"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Nombre maximum de chunks à traiter"
    )
    parser.add_argument(
        "--api-key",
        help="Clé API pour le provider (si non définie dans les variables d'environnement)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Fichier de sortie pour les données extraites (JSON)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Température pour la génération (0.0-1.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher plus de détails sur les résultats"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Extraire également les métadonnées du PDF"
    )
    parser.add_argument(
        "--host",
        help="URL du serveur API (pour lmstudio et ollama)"
    )
    
    # Options spécifiques pour LM Studio et autres modèles
    advanced_group = parser.add_argument_group('Options avancées')
    advanced_group.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Délai d'attente pour les requêtes LLM en secondes"
    )
    advanced_group.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Nombre maximum de tentatives en cas d'erreur"
    )
    
    args = parser.parse_args()
    
    # Joindre tous les arguments de la requête en une seule chaîne
    query = " ".join(args.query)
    
    # Extraire les données
    extract_from_pdf(
        pdf_path=args.pdf_path,
        query=query,
        provider=args.provider,
        model=args.model,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        chunk_method=args.chunk_method,
        api_key=args.api_key,
        output_file=args.output,
        temperature=args.temperature,
        verbose=args.verbose,
        extract_metadata=args.metadata,
        host=args.host,
        timeout=args.timeout,
        max_retries=args.max_retries
    )

if __name__ == "__main__":
    main()
