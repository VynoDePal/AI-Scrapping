#!/usr/bin/env python3
"""
Script CLI pour l'extraction de données structurées avec des LLM.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import des modules du package
from ai_scrapping_toolkit.src.processors import html_to_chunks, pdf_to_chunks
from ai_scrapping_toolkit.src.llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results

def main():
    parser = argparse.ArgumentParser(
        description="Extrait des données structurées à partir de contenu HTML ou PDF en utilisant un modèle de langage"
    )
    parser.add_argument(
        "file_path",
        help="Chemin du fichier HTML ou PDF à analyser"
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Requête d'extraction en langage naturel"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama", "lmstudio", "huggingface", "openrouter"],
        default="openai",
        help="Provider du modèle de langage"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Modèle à utiliser"
    )
    parser.add_argument(
        "--chunk-method",
        choices=["tags", "length", "hybrid", "semantic"],
        default="semantic",
        help="Méthode de chunking"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Taille maximale des chunks"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Nombre maximum de chunks à traiter"
    )
    parser.add_argument(
        "--output", "-o",
        help="Fichier de sortie pour sauvegarder les résultats"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Température pour la génération"
    )
    parser.add_argument(
        "--api-key",
        help="Clé API pour le provider"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher plus de détails"
    )
    
    args = parser.parse_args()
    
    # Joindre tous les arguments de la requête en une seule chaîne
    query = " ".join(args.query)
    
    # Vérifier que le fichier existe
    if not os.path.exists(args.file_path):
        print(f"Erreur: Le fichier {args.file_path} n'existe pas.")
        sys.exit(1)
    
    # Déterminer le type de fichier et extraire les chunks
    file_path = Path(args.file_path)
    if file_path.suffix.lower() == '.pdf':
        logger.info("Traitement d'un fichier PDF")
        chunks = pdf_to_chunks(
            str(file_path),
            method=args.chunk_method,
            max_length=args.chunk_size
        )
    else:
        logger.info("Traitement d'un fichier HTML/texte")
        with open(args.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = html_to_chunks(
            content,
            method=args.chunk_method,
            max_length=args.chunk_size
        )
    
    if not chunks:
        print("Erreur: Aucun chunk généré à partir du fichier.")
        sys.exit(1)
    
    logger.info(f"{len(chunks)} chunks générés")
    
    # Limiter le nombre de chunks si spécifié
    if args.max_chunks and len(chunks) > args.max_chunks:
        chunks = chunks[:args.max_chunks]
        logger.info(f"Limitation à {args.max_chunks} chunks")
    
    # Initialiser le provider LLM
    llm_config = {
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature
    }
    
    try:
        llm_provider = get_llm_provider(args.provider, **llm_config)
    except Exception as e:
        print(f"Erreur lors de l'initialisation du provider LLM: {e}")
        sys.exit(1)
    
    # Extraire les données
    logger.info("Démarrage de l'extraction de données...")
    try:
        extraction_results = extract_data_from_chunks(
            chunks=chunks,
            query=query,
            llm_provider=llm_provider,
            max_workers=min(4, len(chunks))
        )
        
        # Agréger les résultats
        aggregated_data = aggregate_extraction_results(extraction_results)
        
        # Afficher ou sauvegarder les résultats
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
            print(f"Résultats sauvegardés dans {args.output}")
        else:
            print(json.dumps(aggregated_data, indent=2, ensure_ascii=False))
        
        if args.verbose:
            print(f"\nInformations détaillées:")
            print(f"- Chunks traités: {len(chunks)}")
            print(f"- Provider: {args.provider}")
            print(f"- Modèle: {args.model}")
            print(f"- Méthode de chunking: {args.chunk_method}")
    
    except Exception as e:
        print(f"Erreur lors de l'extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
