#!/usr/bin/env python3
# Test de l'environnement pour le projet de scraping avec IA

import sys

def check_imports():
    modules = {
        "requests": "Requêtes HTTP",
        "bs4": "BeautifulSoup pour le parsing HTML",
        "selenium": "Automatisation de navigateur",
        "transformers": "Modèles de langage Hugging Face",
        "sentence_transformers": "Embeddings de phrases",
        "pandas": "Manipulation de données",
        "numpy": "Calcul numérique",
        "faiss": "Recherche vectorielle rapide",
        "nltk": "Traitement du langage naturel",
        "sklearn": "Machine learning"
    }
    
    all_passed = True
    print("Vérification des modules installés:")
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"✓ {module} - {description}")
        except ImportError:
            print(f"❌ {module} - {description} (MANQUANT)")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("Test de l'environnement de développement pour le scraping IA")
    print("-" * 60)
    
    if check_imports():
        print("-" * 60)
        print("✅ Toutes les dépendances sont correctement installées!")
        print("Votre environnement est prêt pour le scraping et l'analyse par IA.")
    else:
        print("-" * 60)
        print("⚠️ Certaines dépendances sont manquantes.")
        print("Veuillez exécuter 'python setup.py' pour installer les dépendances manquantes.")
        sys.exit(1)
