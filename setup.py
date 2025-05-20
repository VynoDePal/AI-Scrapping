#!/usr/bin/env python3
"""
Script de configuration pour l'environnement de scraping web avec IA
"""
import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Vérifie que la version de Python est adéquate."""
    if sys.version_info < (3, 8):
        print("Ce projet nécessite Python 3.8 ou supérieur")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} détecté")

def install_requirements():
    """Installe les bibliothèques requises."""
    print("\nInstallation des dépendances...")
    
    current_dir = Path(__file__).parent
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"Fichier requirements.txt introuvable à {requirements_file}")
        sys.exit(1)
        
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Les dépendances ont été installées avec succès")
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation des dépendances")
        sys.exit(1)

def setup_nltk():
    """Télécharge les ressources NLTK nécessaires."""
    print("\nConfiguration des ressources NLTK...")
    try:
        import nltk
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"✓ Ressource NLTK '{resource}' déjà disponible")
            except LookupError:
                print(f"Téléchargement de la ressource NLTK '{resource}'...")
                nltk.download(resource, quiet=True)
                print(f"✓ Ressource NLTK '{resource}' téléchargée")
    except Exception as e:
        print(f"❌ Erreur lors de la configuration de NLTK: {e}")

def setup_selenium():
    """Configure Selenium et vérifie l'installation du driver."""
    print("\nConfiguration de Selenium...")
    try:
        from selenium import webdriver
        from webdriver_manager.chrome import ChromeDriverManager
        from webdriver_manager.firefox import GeckoDriverManager
        
        print("Téléchargement des drivers pour les navigateurs courants...")
        try:
            ChromeDriverManager().install()
            print("✓ ChromeDriver installé")
        except Exception as e:
            print(f"⚠️ ChromeDriver non installé: {e}")
            
        try:
            GeckoDriverManager().install()
            print("✓ GeckoDriver installé")
        except Exception as e:
            print(f"⚠️ GeckoDriver non installé: {e}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la configuration de Selenium: {e}")

def download_transformer_models():
    """Précharge les modèles transformer de base."""
    print("\nPréchargement des modèles transformer (cela peut prendre un moment)...")
    try:
        # Précharger le modèle de base de Hugging Face
        from transformers import AutoTokenizer, AutoModel
        model_name = "distilbert-base-uncased"
        print(f"Téléchargement du modèle {model_name}...")
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        print(f"✓ Modèle {model_name} téléchargé")
        
        # Précharger le modèle sentence-transformers
        from sentence_transformers import SentenceTransformer
        model_name = "all-MiniLM-L6-v2"
        print(f"Téléchargement du modèle {model_name}...")
        SentenceTransformer(model_name)
        print(f"✓ Modèle {model_name} téléchargé")
    except Exception as e:
        print(f"⚠️ Erreur lors du téléchargement des modèles transformer: {e}")
        print("  Vous pourrez télécharger ces modèles à la demande lors de l'utilisation.")

def create_test_script():
    """Crée un script de test pour vérifier l'installation."""
    test_script_path = Path(__file__).parent / "test_environment.py"
    
    with open(test_script_path, "w") as f:
        f.write("""#!/usr/bin/env python3
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
""")
    
    print(f"\n✓ Script de test créé à {test_script_path}")
    print("  Exécutez 'python test_environment.py' pour vérifier votre installation")

def main():
    """Fonction principale qui exécute toutes les étapes de configuration."""
    print("=" * 70)
    print("CONFIGURATION DE L'ENVIRONNEMENT POUR LE SCRAPING WEB AVEC IA")
    print("=" * 70)
    
    check_python_version()
    install_requirements()
    setup_nltk()
    setup_selenium()
    download_transformer_models()
    create_test_script()
    
    print("\n" + "=" * 70)
    print("CONFIGURATION TERMINÉE!")
    print("=" * 70)
    print("\nPour vérifier l'installation, exécutez: python test_environment.py")
    print("\nCommande pour installer les dépendances manuellement:")
    print(f"pip install -r {Path(__file__).parent}/requirements.txt")

if __name__ == "__main__":
    main()
```

#!/usr/bin/env python3
"""
Configuration pour l'installation du package AI Scrapping Toolkit.
"""

from setuptools import setup, find_packages
import os

# Lire le contenu du README pour la description longue
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Lire les dépendances depuis requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    required_packages = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

# Version du package
version = "1.0.0"

setup(
    name="ai_scrapping_toolkit",
    version=version,
    author="Kevyn Odjo",
    author_email="kevyn.odjo@example.com",
    description="Toolkit pour extraire, prétraiter et traiter des contenus web avec des modèles d'IA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevyn-odjo/ai-scrapping",
    project_urls={
        "Bug Tracker": "https://github.com/kevyn-odjo/ai-scrapping/issues",
        "Documentation": "https://github.com/kevyn-odjo/ai-scrapping",
        "Source Code": "https://github.com/kevyn-odjo/ai-scrapping",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Markup :: HTML",
    ],
    python_requires=">=3.8",
    install_requires=required_packages,
    entry_points={
        "console_scripts": [
            "ai-scraper=ai_scrapping_toolkit.cli.web_scraper:main",
            "ai-search=ai_scrapping_toolkit.cli.search:main",
            "ai-smart-search=ai_scrapping_toolkit.cli.smart_search:main",
            "ai-extract=ai_scrapping_toolkit.cli.extract_data:main",
            "ai-process=ai_scrapping_toolkit.cli.process_data:main",
            "ai-export=ai_scrapping_toolkit.cli.export_data:main",
            "ai-api=ai_scrapping_toolkit.cli.run_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_scrapping_toolkit": ["*.md", "*.txt"],
    },
)
