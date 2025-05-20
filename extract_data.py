#!/usr/bin/env python3
"""
Outil d'extraction de données à partir de chunks HTML en utilisant des modèles de langage.
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any, Optional

from src.llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results
from src.utils.file_handler import load_file
from src.processors import html_to_chunks

def extract_from_file(
    file_path: str,
    query: str,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    chunk_size: int = 4000,
    max_chunks: Optional[int] = None,
    chunk_method: str = "hybrid",
    api_key: Optional[str] = None,
    output_file: Optional[str] = None,
    temperature: float = 0.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Extrait des données d'un fichier HTML en utilisant un modèle de langage.
    
    Args:
        file_path (str): Chemin du fichier HTML
        query (str): Requête d'extraction (ex: "Extraire tous les titres et dates")
        provider (str): Provider du modèle de langage (openai, ollama, huggingface)
        model (str): Modèle à utiliser
        chunk_size (int): Taille maximale des chunks
        max_chunks (int, optional): Nombre maximum de chunks à traiter
        chunk_method (str): Méthode de chunking (tags, length, hybrid)
        api_key (str, optional): Clé API pour le provider
        output_file (str, optional): Fichier de sortie pour les résultats
        temperature (float): Température pour la génération
        verbose (bool): Afficher des informations détaillées
        
    Returns:
        Dict[str, Any]: Données extraites
    """
    # Charger le fichier
    content = load_file(file_path)
    if not content:
        print(f"Erreur: Impossible de charger le fichier {file_path}")
        return {}
    
    # Découper le contenu en chunks
    print(f"Découpage du contenu en chunks (méthode: {chunk_method}, taille max: {chunk_size})...")
    chunks = html_to_chunks(content, method=chunk_method, max_length=chunk_size)
    print(f"{len(chunks)} chunks générés.")
    
    # Limiter le nombre de chunks si spécifié
    if max_chunks and max_chunks > 0 and max_chunks < len(chunks):
        print(f"Limitation à {max_chunks} chunks (sur {len(chunks)} disponibles).")
        chunks = chunks[:max_chunks]
    
    # Initialiser le provider LLM
    llm_config = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature
    }
    
    try:
        llm_provider = get_llm_provider(provider, **llm_config)
    except ImportError as e:
        print(f"Erreur: {str(e)}")
        print("Vérifiez que vous avez installé les dépendances nécessaires.")
        return {}
    
    # Extraire les données des chunks
    print(f"Extraction des données avec {provider} ({model})...")
    print(f"Requête: {query}")
    
    extraction_results = extract_data_from_chunks(
        chunks=chunks,
        query=query,
        llm_provider=llm_provider,
        max_workers=min(4, len(chunks))
    )
    
    # Agréger les résultats
    print("Agrégation des résultats...")
    aggregated_data = aggregate_extraction_results(extraction_results)
    
    # Afficher les statistiques
    for data_type, items in aggregated_data.items():
        print(f"- {data_type}: {len(items)} éléments extraits")
    
    # Sauvegarder les résultats si demandé
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
        print(f"Résultats sauvegardés dans {output_file}")
    
    # Afficher plus d'informations si verbose
    if verbose:
        print("\n=== Détails des résultats ===")
        for data_type, items in aggregated_data.items():
            print(f"\n{data_type.upper()}:")
            for i, item in enumerate(items[:5], 1):
                print(f"  {i}. {json.dumps(item, ensure_ascii=False)}")
            if len(items) > 5:
                print(f"  ... et {len(items) - 5} autres éléments")
    
    return aggregated_data

def main():
    parser = argparse.ArgumentParser(
        description="Extrait des données structurées à partir de contenu HTML en utilisant un modèle de langage"
    )
    parser.add_argument(
        "file_path",
        help="Chemin du fichier HTML à analyser"
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Requête d'extraction en langage naturel (ex: 'Extraire tous les titres et dates')"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama", "huggingface", "lmstudio", "openrouter"],  # Ajout d'OpenRouter
        default="openai",
        help="Provider du modèle de langage"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Modèle à utiliser (ex: gpt-3.5-turbo, llama2, mistral, openrouter: anthropic/claude-3-sonnet)"
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
        default=None,
        help="Nombre maximum de chunks à traiter"
    )
    parser.add_argument(
        "--chunk-method",
        choices=["tags", "length", "hybrid"],
        default="hybrid",
        help="Méthode de découpage en chunks"
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
        "--host",
        help="URL du serveur API (pour lmstudio et ollama)"
    )
    
    # Options spécifiques pour LM Studio
    lmstudio_group = parser.add_argument_group('Options LM Studio')
    lmstudio_group.add_argument(
        "--lmstudio-port", 
        type=int, 
        default=1234,
        help="Port du serveur LM Studio (défaut: 1234)"
    )
    lmstudio_group.add_argument(
        "--retry", 
        action="store_true",
        help="En cas d'erreur, réessayer avec des paramètres plus simples"
    )
    lmstudio_group.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Délai d'attente en secondes pour les requêtes API (défaut: 180)"
    )
    lmstudio_group.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Nombre maximum de tentatives en cas d'erreur (défaut: 3)"
    )
    lmstudio_group.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Délai en secondes entre les tentatives (défaut: 5)"
    )
    
    # Options spécifiques pour OpenRouter
    openrouter_group = parser.add_argument_group('Options OpenRouter')
    openrouter_group.add_argument(
        "--openrouter-key", 
        help="Clé API OpenRouter (ou définir la variable d'environnement OPENROUTER_API_KEY)"
    )
    openrouter_group.add_argument(
        "--max-tokens", 
        type=int, 
        default=2048,
        help="Nombre maximum de tokens à générer"
    )
    
    args = parser.parse_args()
    
    # Joindre tous les arguments de la requête en une seule chaîne
    query = " ".join(args.query)
    
    # Vérifier que le fichier existe
    if not os.path.exists(args.file_path):
        print(f"Erreur: Le fichier {args.file_path} n'existe pas.")
        sys.exit(1)
    
    # Initialiser le provider LLM - CORRECTION : définir llm_config d'abord
    llm_config = {
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature
    }
    
    # Configuration spécifique pour LM Studio
    if args.provider == "lmstudio":
        if not args.host:
            args.host = f"http://localhost:{args.lmstudio_port}/v1"
            print(f"Utilisation de l'URL LM Studio par défaut: {args.host}")
        
        # Ajouter les options de timeout et de retry au config
        llm_config["timeout"] = args.timeout
        llm_config["max_retries"] = args.max_retries
        llm_config["retry_delay"] = args.retry_delay
        
        print("⚠️ Important pour LM Studio: ")
        print("1. Assurez-vous que le serveur LM Studio est actif (local server → enable)")
        print(f"2. Vérifiez que le modèle est chargé et que le port {args.lmstudio_port} est correct")
        print("3. Pour les longs textes, l'opération peut prendre plusieurs minutes")
        print(f"4. Timeout configuré à {args.timeout}s avec {args.max_retries} tentatives maximum")
    
    # Configuration spécifique pour OpenRouter
    elif args.provider == "openrouter":
        if args.openrouter_key:
            llm_config["api_key"] = args.openrouter_key
        
        llm_config["max_tokens"] = args.max_tokens
        llm_config["timeout"] = args.timeout if hasattr(args, 'timeout') else 120
        
        print("⚠️ Important pour OpenRouter: ")
        print("1. Assurez-vous d'avoir défini une clé API OpenRouter valide")
        print("2. Format des modèles: 'provider/model' (ex: 'anthropic/claude-3-sonnet')")
        print("3. Liste des modèles disponibles: https://openrouter.ai/docs#models")
        print(f"4. Modèle sélectionné: {args.model}")
    
    # Ajouter l'URL du serveur si spécifiée
    if args.host:
        llm_config["host"] = args.host
    
    try:
        llm_provider = get_llm_provider(args.provider, **llm_config)
    except ImportError as e:
        print(f"Erreur: {str(e)}")
        print("Vérifiez que vous avez installé les dépendances nécessaires.")
        return {}
    
    # Extraire les données
    extract_from_file(
        file_path=args.file_path,
        query=query,
        provider=args.provider,
        model=args.model,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        chunk_method=args.chunk_method,
        api_key=args.api_key,
        output_file=args.output,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    # Ajouter un bloc de gestion des erreurs et retry pour LM Studio
    if args.provider == "lmstudio" and args.retry:
        try:
            # Extraire les données
            extract_from_file(
                file_path=args.file_path,
                query=query,
                provider=args.provider,
                model=args.model,
                chunk_size=args.chunk_size,
                max_chunks=args.max_chunks,
                chunk_method=args.chunk_method,
                api_key=args.api_key,
                output_file=args.output,
                temperature=args.temperature,
                verbose=args.verbose
            )
        except Exception as e:
            print(f"Erreur lors de l'extraction avec LM Studio: {e}")
            print("Tentative avec des paramètres simplifiés...")
            
            # Simplifier l'instruction pour aider le modèle
            simple_query = f"Extrait seulement les {text_field} du texte suivant au format JSON."
            
            # Réduire la taille des chunks
            simpler_chunks = []
            for chunk in chunks:
                if len(chunk) > args.chunk_size // 2:
                    parts = [chunk[i:i+args.chunk_size//2] for i in range(0, len(chunk), args.chunk_size//2)]
                    simpler_chunks.extend(parts)
                else:
                    simpler_chunks.append(chunk)
            
            # Nouvelle tentative avec les paramètres simplifiés
            extraction_results = extract_data_from_chunks(
                chunks=simpler_chunks[:args.max_chunks if args.max_chunks else len(simpler_chunks)],
                query=simple_query,
                llm_provider=llm_provider,
                max_workers=min(2, len(chunks))
            )
    
    # En cas d'erreur avec LM Studio, offrir des conseils de débogage
    if args.provider == "lmstudio" and not extraction_results:
        print("\n⚠️ Erreur avec LM Studio. Conseils de dépannage:")
        print("1. Vérifiez que le serveur local est activé dans LM Studio")
        print("2. Essayez d'augmenter le timeout: --timeout 300")
        print("3. Réduisez la taille des chunks: --chunk-size 2000")
        print("4. Utilisez un modèle plus performant dans LM Studio")
        print("5. Vérifiez l'adresse du serveur avec --host http://localhost:1234/v1")
        print("6. Consultez les logs pour plus de détails")

if __name__ == "__main__":
    main()
