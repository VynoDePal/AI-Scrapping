#!/usr/bin/env python3
"""
Outil de recherche sémantique dans les index vectoriels FAISS.
"""

import argparse
import os
import sys
from src.embeddings import load_faiss_index, search_similar

def main():
    parser = argparse.ArgumentParser(description="Rechercher dans une base de données vectorielle")
    parser.add_argument("index_path", help="Chemin de base de l'index FAISS")
    parser.add_argument("query", nargs="+", help="Texte de la requête de recherche")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Nombre de résultats à retourner")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Modèle sentence-transformers à utiliser")
    
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
        
        # Effectuer la recherche
        print(f"Recherche: \"{query}\"")
        results = search_similar(query, index, index_metadata, 
                                model_name=args.model, top_k=args.top_k)
        
        # Afficher les résultats
        print(f"\n{len(results)} résultats trouvés:")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.4f}")
            print(f"   Index: {result['index']}")
            
            # Afficher les métadonnées si présentes
            if result['metadata']:
                print(f"   Métadonnées: {result['metadata']}")
                
            # Affichage du chunk (limité pour la lisibilité)
            chunk_text = result['chunk']
            if len(chunk_text) > 300:
                chunk_text = chunk_text[:297] + "..."
            print(f"   Texte: {chunk_text}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Erreur lors de la recherche: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
