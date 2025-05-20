#!/usr/bin/env python3
"""
Outil de recherche intelligent qui analyse les requêtes utilisateur
en langage naturel et renvoie les chunks les plus pertinents.
"""

import argparse
import os
import sys
import json
from src.embeddings import load_faiss_index
from src.nlp import search_with_query, analyze_query

def main():
    parser = argparse.ArgumentParser(
        description="Recherche intelligente avec analyse de requêtes en langage naturel"
    )
    parser.add_argument(
        "index_path", 
        help="Chemin de base de l'index FAISS"
    )
    parser.add_argument(
        "query", 
        nargs="+", 
        help="Requête en langage naturel (ex: 'extraire tous les titres et dates des articles')"
    )
    parser.add_argument(
        "--top-k", "-k", 
        type=int, 
        default=5,
        help="Nombre de résultats à retourner"
    )
    parser.add_argument(
        "--model", 
        default="all-MiniLM-L6-v2",
        help="Modèle sentence-transformers à utiliser"
    )
    parser.add_argument(
        "--advanced", "-a", 
        action="store_true",
        help="Utiliser l'analyse NLP avancée avec transformers"
    )
    parser.add_argument(
        "--filter", "-f", 
        action="store_true",
        help="Filtrer les résultats en fonction des entités détectées"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Fichier de sortie pour les résultats (format JSON)"
    )
    
    args = parser.parse_args()
    
    # Joindre tous les arguments de la requête en une seule chaîne
    query = " ".join(args.query)
    
    # Vérifier que les fichiers d'index existent
    index_path = args.index_path.rstrip(".index").rstrip(".meta")
    if not os.path.exists(f"{index_path}.index"):
        print(f"Erreur: Le fichier d'index '{index_path}.index' n'existe pas.")
        sys.exit(1)
        
    if not os.path.exists(f"{index_path}.meta"):
        print(f"Erreur: Le fichier de métadonnées '{index_path}.meta' n'existe pas.")
        sys.exit(1)
    
    try:
        # Charger l'index FAISS
        print(f"Chargement de l'index vectoriel depuis '{index_path}'...")
        index, index_metadata = load_faiss_index(index_path)
        
        # Analyser la requête et effectuer la recherche
        print(f"Analyse de la requête: \"{query}\"")
        print("=" * 80)
        
        # Analyser la requête pour afficher les entités et intentions détectées
        analysis = analyze_query(query, use_transformers=args.advanced)
        
        # Afficher les résultats de l'analyse
        print("Analyse de la requête:")
        print(f"- Intention détectée: {analysis['intent']} (confiance: {analysis['intent_confidence']:.2f})")
        
        if analysis['entities']:
            print("- Entités détectées:")
            for entity in analysis['entities']:
                print(f"  * {entity['type']}: {entity['value']} (confiance: {entity['confidence']:.2f})")
        else:
            print("- Aucune entité spécifique détectée")
            
        print("\nRecherche des chunks les plus pertinents...")
        
        # Effectuer la recherche
        results = search_with_query(
            query=query,
            index=index,
            index_metadata=index_metadata,
            model_name=args.model,
            top_k=args.top_k,
            use_transformers=args.advanced,
            filter_by_entities=args.filter
        )
        
        # Afficher les résultats
        print(f"\n{len(results)} résultats trouvés:")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.4f}")
            
            if 'original_score' in result:
                bonus = result['score'] - result['original_score']
                print(f"   Score initial: {result['original_score']:.4f}, bonus entités: +{bonus:.4f}")
                
            print(f"   Index: {result['index']}")
            
            # Afficher les métadonnées si présentes
            if result.get('metadata'):
                print(f"   Métadonnées: {result['metadata']}")
                
            # Affichage du chunk (limité pour la lisibilité)
            chunk_text = result['chunk']
            if len(chunk_text) > 300:
                chunk_text = chunk_text[:297] + "..."
            print(f"   Texte: {chunk_text}")
            print("-" * 80)
            
        # Sauvegarder les résultats si demandé
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'analysis': analysis,
                    'results': results
                }, f, ensure_ascii=False, indent=2)
            print(f"\nRésultats sauvegardés dans {args.output}")
                
    except Exception as e:
        print(f"Erreur lors de la recherche: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
