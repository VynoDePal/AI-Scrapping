#!/usr/bin/env python3
"""
Script principal pour récupérer le contenu HTML de sites web.
Fournit une interface en ligne de commande et coordonne le flux de travail.
"""

import argparse
import os
import sys
import traceback
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import des modules personnalisés
from ai_scrapping_toolkit.src.scrapers import fetch_content
from ai_scrapping_toolkit.src.processors import preprocess_html, extract_main_content, get_page_title, html_to_chunks
from ai_scrapping_toolkit.src.utils import save_chunks
from ai_scrapping_toolkit.src.embeddings import chunks_to_embeddings, create_faiss_index, save_faiss_index

def process_content(url, method="auto", wait_time=5, save_to_file=None, preprocess=False, 
                   extract_main=False, chunk=False, chunk_method='hybrid', 
                   max_chunk_length=1000, chunk_overlap=100, vectorize=False,
                   model_name='all-MiniLM-L6-v2', index_type='L2', debug=False,
                   respect_robots=True, user_agent=None, rate_limit=1.0):
    """
    Récupère et traite le contenu HTML d'un site web selon les options spécifiées.
    """
    try:
        # Récupération du contenu HTML avec les paramètres éthiques et légaux
        logger.info(f"Récupération du contenu depuis {url} avec méthode {method}...")
        html_content = fetch_content(
            url, 
            method, 
            wait_time,
            respect_robots=respect_robots,
            user_agent=user_agent,
            rate_limit=rate_limit
        )
        
        if not html_content:
            logger.error("Aucun contenu HTML n'a été récupéré")
            return None
        
        logger.info(f"Contenu récupéré avec succès ({len(html_content)} caractères)")
        
        # Traitement du contenu selon les options
        processed_content = html_content
        
        if extract_main:
            logger.info("Extraction du contenu principal...")
            processed_content = extract_main_content(html_content)
            if not processed_content:
                logger.warning("L'extraction du contenu principal n'a retourné aucun résultat, utilisation du contenu complet")
                processed_content = html_content
            else:
                logger.info(f"Contenu principal extrait ({len(processed_content)} caractères)")
        elif preprocess:
            logger.info("Prétraitement du contenu HTML...")
            processed_content = preprocess_html(html_content)
            logger.info(f"Contenu prétraité ({len(processed_content)} caractères)")
        
        # Découpage en chunks si demandé
        if chunk and processed_content:
            try:
                logger.info(f"Découpage du contenu en chunks (méthode: {chunk_method}, taille max: {max_chunk_length}, chevauchement: {chunk_overlap})...")
                chunks = html_to_chunks(
                    processed_content, 
                    method=chunk_method,
                    max_length=max_chunk_length,
                    overlap=chunk_overlap
                )
                
                if not chunks:
                    logger.error("Le découpage en chunks n'a produit aucun résultat")
                    return None
                
                logger.info(f"{len(chunks)} chunks générés")
                
                # Vectorisation des chunks si demandée
                if vectorize and chunks:
                    try:
                        logger.info(f"Vectorisation des chunks avec le modèle {model_name}...")
                        
                        # Génération des embeddings
                        embeddings = chunks_to_embeddings(chunks, model_name=model_name)
                        
                        # Création de l'index FAISS
                        index, index_metadata = create_faiss_index(embeddings, chunks, index_type)
                        
                        # Sauvegarde de l'index si un fichier de sortie est spécifié
                        if save_to_file:
                            base_path = save_to_file.rsplit('.', 1)[0] if '.' in save_to_file else save_to_file
                            index_path = f"{base_path}_vectordb"
                            save_faiss_index(index, index_metadata, index_path)
                        
                        # Continuer avec le traitement des chunks
                        if save_to_file and chunks:
                            save_chunks(chunks, processed_content, save_to_file)
                        
                        # Retourner un tuple avec les chunks et l'index FAISS
                        return chunks, index, index_metadata
                    
                    except Exception as e:
                        logger.error(f"Erreur lors de la vectorisation: {e}")
                        if debug:
                            traceback.print_exc()
                        # En cas d'erreur, continuer normalement avec les chunks
                
                # Sauvegarde des chunks si nécessaire
                if save_to_file and chunks:
                    save_chunks(chunks, processed_content, save_to_file)
                
                return chunks
            except Exception as e:
                logger.error(f"Erreur lors du découpage en chunks: {e}")
                if debug:
                    traceback.print_exc()
                return None
        
        # Sinon, sauvegarde du contenu standard si nécessaire
        elif save_to_file and processed_content:
            from src.utils.file_handler import save_file
            save_file(processed_content, save_to_file)
            logger.info(f"Contenu {'prétraité' if (preprocess or extract_main) else 'HTML'} sauvegardé dans {save_to_file}")
        
        return processed_content
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du contenu: {e}")
        if debug:
            traceback.print_exc()
        return None

def display_result(result, show_title=False, preprocess=False, extract_main=False, output=None, vectorize=False):
    """
    Affiche des informations sur le résultat du scraping.
    """
    if not result:
        print("Échec de la récupération du contenu.")
        return
    
    # Cas où le résultat est un tuple avec l'index vectoriel
    if vectorize and isinstance(result, tuple) and len(result) == 3:
        chunks, index, index_metadata = result
        print(f"\nRécupération et vectorisation réussies!")
        print(f"- {len(chunks)} chunks générés et vectorisés")
        print(f"- Index vectoriel créé avec {index.ntotal} vecteurs de dimension {index_metadata['dim']}")
        
        if not output:
            preview_count = min(3, len(chunks))
            for i in range(preview_count):
                chunk_preview = chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i]
                print(f"\nChunk {i+1}/{len(chunks)} (aperçu): {chunk_preview}")
            print("\nUtilisez l'option --output pour sauvegarder tous les chunks et l'index vectoriel.")
        return
        
    # Cas standard
    if isinstance(result, list):  # Résultat en chunks
        print(f"\nRécupération réussie! {len(result)} chunks générés.")
        if not output:
            preview_count = min(3, len(result))
            for i in range(preview_count):
                chunk_preview = result[i][:100] + "..." if len(result[i]) > 100 else result[i]
                print(f"\nChunk {i+1}/{len(result)} (aperçu): {chunk_preview}")
            print("\nUtilisez l'option --output pour sauvegarder tous les chunks.")
    else:  # Résultat en texte brut ou HTML
        if show_title and not preprocess and not extract_main:
            title = get_page_title(result)
            print(f"\nTitre de la page: {title}")
        
        content_type = "texte prétraité" if preprocess or extract_main else "HTML"
        print(f"\nRécupération réussie! Longueur du {content_type}: {len(result)} caractères")
        
        if not output:
            if preprocess or extract_main:
                preview_length = min(150, len(result))
                print(f"\nAperçu: \n{result[:preview_length]}...")
            print("Utilisez l'option --output pour sauvegarder le contenu complet.")

def main():
    parser = argparse.ArgumentParser(description="Récupérer et traiter le contenu HTML d'un site web")
    parser.add_argument("url", help="URL du site à scraper")
    
    # Options de récupération
    parser.add_argument("--method", choices=["requests", "selenium", "auto"], 
                        default="auto", help="Méthode à utiliser (par défaut: auto)")
    parser.add_argument("--wait", type=int, default=5,
                        help="Temps d'attente après chargement pour Selenium (secondes)")
    
    # Options de traitement
    parser.add_argument("--output", "-o", help="Nom du fichier pour sauvegarder le contenu")
    parser.add_argument("--show-title", action="store_true", help="Afficher le titre de la page")
    parser.add_argument("--preprocess", action="store_true",
                        help="Prétraiter le HTML pour extraire uniquement le texte pertinent")
    parser.add_argument("--main-content", action="store_true",
                        help="Extraire uniquement le contenu principal de la page")
    
    # Options pour le chunking
    parser.add_argument("--chunk", action="store_true",
                        help="Diviser le contenu en segments (chunks)")
    parser.add_argument("--chunk-method", choices=["tags", "length", "hybrid", "semantic"],
                        default="hybrid", help="Méthode de découpage en chunks")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Taille maximale d'un chunk en caractères")
    parser.add_argument("--chunk-overlap", type=int, default=100,
                        help="Chevauchement entre chunks en caractères")
    
    # Nouvelles options pour la vectorisation
    parser.add_argument("--vectorize", action="store_true",
                        help="Vectoriser les chunks avec un modèle d'embedding")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Modèle sentence-transformers à utiliser pour la vectorisation")
    parser.add_argument("--index-type", choices=["L2", "IP", "IVF"], default="L2",
                        help="Type d'index FAISS à créer")
    
    # Options éthiques et légales
    legal_group = parser.add_argument_group('Options Légales & Éthiques')
    legal_group.add_argument("--ignore-robots", action="store_true",
                        help="Ignorer les règles robots.txt (non recommandé)")
    legal_group.add_argument("--user-agent", 
                        default="AI-Scrapping-Toolkit/1.0 (+https://github.com/kevyn-odjo/ai-scrapping)",
                        help="User-Agent à utiliser pour les requêtes")
    legal_group.add_argument("--rate-limit", type=float, default=1.0,
                        help="Délai minimum entre les requêtes en secondes")
    
    # Options de débogage
    parser.add_argument("--debug", action="store_true", 
                        help="Activer le mode débogage pour afficher plus d'informations")
    
    args = parser.parse_args()
    
    # Configurer le niveau de logging selon le mode debug
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Vérification de cohérence des arguments
    if args.vectorize and not args.chunk:
        logger.info("L'option --vectorize nécessite l'option --chunk. Activation automatique du chunking.")
        args.chunk = True
    
    # Traitement du contenu avec les paramètres éthiques et légaux
    result = process_content(
        args.url,
        method=args.method,
        wait_time=args.wait,
        save_to_file=args.output,
        preprocess=args.preprocess,
        extract_main=args.main_content,
        chunk=args.chunk,
        chunk_method=args.chunk_method,
        max_chunk_length=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vectorize=args.vectorize,
        model_name=args.model,
        index_type=args.index_type,
        debug=args.debug,
        respect_robots=not args.ignore_robots,
        user_agent=args.user_agent,
        rate_limit=args.rate_limit
    )
    
    # Affichage des résultats
    display_result(
        result,
        show_title=args.show_title,
        preprocess=args.preprocess,
        extract_main=args.main_content,
        output=args.output,
        vectorize=args.vectorize
    )

if __name__ == "__main__":
    main()