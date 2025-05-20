#!/usr/bin/env python3
"""
Script pour traiter des données extraites selon les préférences de l'utilisateur.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Any, Optional

from src.processors.data_processor import (
    filter_by_date, 
    analyze_sentiment, 
    categorize_text, 
    sort_and_filter
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Charge des données depuis un fichier JSON.
    
    Args:
        file_path (str): Chemin du fichier JSON
        
    Returns:
        Dict: Données chargées
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier JSON '{file_path}': {str(e)}")
        sys.exit(1)

def save_json_data(data: Dict[str, Any], file_path: str) -> None:
    """
    Sauvegarde des données dans un fichier JSON.
    
    Args:
        data (Dict): Données à sauvegarder
        file_path (str): Chemin du fichier de destination
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Données sauvegardées dans '{file_path}'")
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement dans '{file_path}': {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Traite des données extraites selon les préférences de l'utilisateur"
    )
    parser.add_argument(
        "input_file",
        help="Fichier JSON contenant les données extraites"
    )
    parser.add_argument(
        "--output", "-o",
        help="Fichier de sortie pour les données traitées"
    )
    
    # Options de filtrage par date
    date_group = parser.add_argument_group('Filtrage par date')
    date_group.add_argument(
        "--filter-date",
        action="store_true",
        help="Filtrer les données par date"
    )
    date_group.add_argument(
        "--date-field",
        default="date",
        help="Nom du champ contenant la date (par défaut: 'date')"
    )
    date_group.add_argument(
        "--days",
        type=int,
        default=30,
        help="Nombre de jours à considérer (par défaut: 30 derniers jours)"
    )
    date_group.add_argument(
        "--start-date",
        help="Date de début au format YYYY-MM-DD"
    )
    date_group.add_argument(
        "--end-date",
        help="Date de fin au format YYYY-MM-DD"
    )
    
    # Options d'analyse de sentiment
    sentiment_group = parser.add_argument_group('Analyse de sentiment')
    sentiment_group.add_argument(
        "--analyze-sentiment",
        action="store_true",
        help="Analyser le sentiment des textes"
    )
    sentiment_group.add_argument(
        "--sentiment-field",
        default="titre",
        help="Nom du champ contenant le texte à analyser (par défaut: 'titre')"
    )
    sentiment_group.add_argument(
        "--sentiment-model",
        default="nlptown/bert-base-multilingual-uncased-sentiment",
        help="Nom du modèle à utiliser pour l'analyse de sentiment"
    )
    sentiment_group.add_argument(
        "--sentiment-provider",
        choices=["huggingface", "openai", "ollama"],
        default="huggingface",
        help="Provider pour l'analyse de sentiment"
    )
    
    # Options de catégorisation
    category_group = parser.add_argument_group('Catégorisation')
    category_group.add_argument(
        "--categorize",
        action="store_true",
        help="Catégoriser les textes"
    )
    category_group.add_argument(
        "--category-field",
        default="titre",
        help="Nom du champ contenant le texte à catégoriser (par défaut: 'titre')"
    )
    category_group.add_argument(
        "--categories",
        help="Liste de catégories séparées par des virgules"
    )
    category_group.add_argument(
        "--category-model",
        default="facebook/bart-large-mnli",
        help="Nom du modèle à utiliser pour la catégorisation"
    )
    category_group.add_argument(
        "--category-provider",
        choices=["huggingface", "openai", "ollama"],
        default="huggingface",
        help="Provider pour la catégorisation"
    )
    
    # Options de tri et filtrage
    sort_group = parser.add_argument_group('Tri et filtrage')
    sort_group.add_argument(
        "--sort-by",
        help="Champ pour le tri (ex: 'date', 'sentiment_score')"
    )
    sort_group.add_argument(
        "--sort-desc",
        action="store_true",
        help="Trier par ordre décroissant"
    )
    sort_group.add_argument(
        "--filter",
        help="Expression de filtrage (ex: \"sentiment == 'positif'\")"
    )
    
    args = parser.parse_args()
    
    # Charger les données
    data = load_json_data(args.input_file)
    
    # Traiter les données
    processed_data = data
    
    # 1. Filtrer par date si demandé
    if args.filter_date:
        logger.info(f"Filtrage par date (champ: {args.date_field}, période: {args.days} jours)")
        processed_data = filter_by_date(
            processed_data,
            date_field=args.date_field,
            days=args.days,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    # 2. Analyser le sentiment si demandé
    if args.analyze_sentiment:
        logger.info(f"Analyse de sentiment (champ: {args.sentiment_field}, provider: {args.sentiment_provider})")
        processed_data = analyze_sentiment(
            processed_data,
            text_field=args.sentiment_field,
            model_name=args.sentiment_model,
            provider=args.sentiment_provider
        )
    
    # 3. Catégoriser si demandé
    if args.categorize:
        categories = None
        if args.categories:
            categories = [cat.strip() for cat in args.categories.split(',')]
        
        logger.info(f"Catégorisation (champ: {args.category_field}, provider: {args.category_provider})")
        processed_data = categorize_text(
            processed_data,
            text_field=args.category_field,
            categories=categories,
            model_name=args.category_model,
            provider=args.category_provider
        )
    
    # 4. Trier et filtrer si demandé
    if args.sort_by or args.filter:
        logger.info(f"Tri et filtrage (tri par: {args.sort_by or 'aucun'}, filtre: {args.filter or 'aucun'})")
        processed_data = sort_and_filter(
            processed_data,
            sort_by=args.sort_by,
            ascending=not args.sort_desc,
            filter_expr=args.filter
        )
    
    # Afficher des informations sur les données traitées
    for key, value in processed_data.items():
        if isinstance(value, list):
            logger.info(f"- {key}: {len(value)} éléments")
    
    # Sauvegarder les données traitées
    if args.output:
        save_json_data(processed_data, args.output)
    else:
        # Afficher un aperçu des données
        print("\nAperçu des données traitées:")
        for key, value in processed_data.items():
            if isinstance(value, list) and value:
                print(f"\n{key.upper()}:")
                for item in value[:3]:
                    print(f"  - {item}")
                if len(value) > 3:
                    print(f"  ... et {len(value) - 3} autres éléments")

if __name__ == "__main__":
    main()
