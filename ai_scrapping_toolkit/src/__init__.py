"""
Module d'initialisation du projet AI Scrapping.
"""

import os
from pathlib import Path

# Chargement des variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    # Chemin vers le fichier .env à la racine du projet
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # Vérifier si les clés API principales sont disponibles
    api_keys = {
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'OPENROUTER_API_KEY': os.environ.get('OPENROUTER_API_KEY'),
        'HUGGINGFACE_API_KEY': os.environ.get('HUGGINGFACE_API_KEY')
    }
    
    loaded_keys = [key for key, value in api_keys.items() if value]
    if loaded_keys:
        print(f"Variables d'environnement chargées: {', '.join(loaded_keys)}")
    else:
        print("Aucune clé API trouvée dans le fichier .env")
    
except ImportError:
    print("Le package python-dotenv n'est pas installé. Les variables d'environnement .env ne seront pas chargées.")
    print("Pour l'installer: pip install python-dotenv")
except Exception as e:
    print(f"Erreur lors du chargement des variables d'environnement: {e}")

__version__ = "1.0.0"
