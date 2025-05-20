"""
AI Scrapping Toolkit - Bibliothèque pour extraire, prétraiter et traiter des contenus web avec des modèles d'IA.
"""

__version__ = "1.0.0"
__author__ = "Kevyn Odjo"

# Import des modules principaux pour faciliter l'accès
from .src.scrapers import fetch_content
from .src.processors import preprocess_html, extract_main_content, html_to_chunks, pdf_to_chunks
from .src.embeddings import chunks_to_embeddings, create_faiss_index
from .src.llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results

# Configurer le logger
import logging
import os

# Tenter de charger les variables d'environnement
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info(f"Variables d'environnement chargées depuis {env_path}")
except ImportError:
    pass

# Configuration du logger par défaut
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
