# AI Scraping Toolkit

Toolkit pour extraire, prétraiter et traiter des contenus web avec des modèles d'IA.

## Installation

### Installation depuis PyPI (recommandé)

```bash
pip install ai-scrapping-toolkit
```

### Installation depuis le dépôt GitHub

```bash
pip install git+https://github.com/kevyn-odjo/ai-scrapping.git
```

### Installation pour le développement

```bash
git clone https://github.com/kevyn-odjo/ai-scrapping.git
cd ai-scrapping
pip install -e .
```

## Configuration des clés API

Pour utiliser les modèles de langage nécessitant une authentification, vous devez configurer vos clés API:

1. Copiez le fichier `.env.example` vers `.env`:
   ```bash
   cp .env.example .env
   ```

2. Éditez le fichier `.env` et ajoutez vos clés API:
   ```bash
   nano .env
   ```
   
3. Les clés API suivantes sont prises en charge:
   - `OPENAI_API_KEY`: Pour les modèles GPT d'OpenAI
   - `OPENROUTER_API_KEY`: Pour accéder à divers modèles via OpenRouter
   - `HUGGINGFACE_API_KEY`: Pour les API Hugging Face Inference

### Sécurité des clés API
- Ne partagez jamais votre fichier `.env` (il est ajouté à `.gitignore` par défaut)
- Vous pouvez aussi définir ces variables d'environnement directement dans votre système

## Fonctionnalités principales

### 1. Récupération de contenu web

```bash
python web_scraper.py example.com
```

Options:
- `--method` : Méthode de récupération (`requests`, `selenium`, `auto`)
- `--wait` : Temps d'attente pour Selenium (secondes)
- `--output` ou `-o` : Fichier de sortie
- `--show-title` : Affiche le titre de la page

Options Légales & Éthiques:
- `--ignore-robots` : Désactive la vérification des règles robots.txt (non recommandé)
- `--user-agent` : Spécifie un User-Agent personnalisé pour les requêtes
- `--rate-limit` : Définit le délai minimum entre les requêtes en secondes (défaut: 1.0s)

```bash
# Exemple de scraping respectueux avec rate limiting
python web_scraper.py example.com --rate-limit 2.5 --output resultat.html
```

### 2. Prétraitement et extraction du contenu principal

```bash
# Prétraiter le HTML pour obtenir du texte propre
python web_scraper.py example.com --preprocess -o texte_propre.txt

# Extraire uniquement le contenu principal de la page
python web_scraper.py example.com --main-content -o contenu_principal.txt
```

### 3. Segmentation en chunks pour les modèles d'IA

```bash
# Diviser le contenu en chunks par méthode hybride
python web_scraper.py example.com --preprocess --chunk -o sortie.txt

# Personnaliser la méthode et taille des chunks
python web_scraper.py example.com --chunk --chunk-method tags --chunk-size 800 --chunk-overlap 50 -o sortie.txt

# Utiliser le mode debug pour diagnostiquer les problèmes
python web_scraper.py example.com --preprocess --chunk --debug -o sortie.txt
```

Méthodes de chunking:
- `tags`: Découpage basé sur les balises HTML (p, div, headings...)
- `length`: Découpage par longueur fixe (plus robuste pour les sites avec peu de structure)
- `hybrid`: Combinaison des deux approches (recommandé)

En cas d'erreur avec l'option `--chunk`, vous pouvez:
1. Essayer la méthode `length` qui est plus robuste: `--chunk-method length`
2. Augmenter la taille des chunks: `--chunk-size 2000`
3. Activer le mode debug pour plus d'informations: `--debug`

### 4. Vectorisation et recherche sémantique

```bash
# Récupérer, découper et vectoriser le contenu d'un site web
python web_scraper.py example.com --preprocess --chunk --vectorize -o data/exemple

# Rechercher dans une base vectorielle existante
python search.py data/exemple_vectordb "ma requête de recherche" --top-k 5
```

Options de vectorisation:
- `--vectorize` : Active la vectorisation des chunks
- `--model` : Modèle sentence-transformers à utiliser (par défaut: all-MiniLM-L6-v2)
- `--index-type` : Type d'index FAISS (L2, IP, IVF)

### 5. Recherche intelligente avec analyse NLP

La fonctionnalité d'analyse NLP permet de comprendre les requêtes en langage naturel et d'optimiser les résultats:

```bash
# Recherche intelligente qui analyse la requête
python smart_search.py data/exemple_vectordb "extraire tous les titres et dates des articles"

# Utilisation de l'analyse NLP avancée avec filtrage par entités
python smart_search.py data/exemple_vectordb "trouver les prix des produits" --advanced --filter
```

### 6. Extraction de données structurées avec LLM

Nouvelle fonctionnalité! Extraire des données structurées à partir du contenu HTML en utilisant des modèles de langage:

```bash
# Extraire les titres et dates d'un fichier HTML avec OpenAI
python extract_data.py contenu.html "Extraire tous les titres et dates des articles" --output resultat.json

# Utiliser Ollama (modèle local) pour l'extraction
python extract_data.py contenu.html "Extraire les prix et descriptions des produits" --provider ollama --model mistral

# Utiliser LM Studio avec options avancées
python extract_data.py contenu.html "Extraire les informations de contact" --provider lmstudio --timeout 300 --max-retries 5

# Utiliser OpenRouter pour accéder à divers modèles (Claude, etc.)
python extract_data.py contenu.html "Extraire les spécifications techniques" --provider openrouter --model anthropic/claude-3-sonnet
```

Options d'extraction:
- `--provider`: Modèle de langage à utiliser (`openai`, `ollama`, `lmstudio`, `huggingface`, `openrouter`)
- `--model`: Nom du modèle spécifique (ex: `gpt-3.5-topenaiurbo`, `llama2`, `mistral`, `anthropic/claude-3-sonnet`)
- `--host`: URL du serveur API (pour `lmstudio` et `ollama`)
- `--chunk-size`: Taille maximale des chunks à envoyer au modèle
- `--max-chunks`: Limiter le nombre de chunks à traiter (utile pour les tests)
- `--output` ou `-o`: Sauvegarder les résultats dans un fichier JSON
- `--verbose` ou `-v`: Afficher plus de détails sur les résultats

Options spécifiques pour LM Studio:
- `--lmstudio-port`: Port du serveur LM Studio (défaut: 1234)
- `--retry`: En cas d'erreur, réessayer avec des paramètres simplifiés
- `--timeout`: Délai d'attente en secondes pour les requêtes API (défaut: 180)
- `--max-retries`: Nombre maximum de tentatives en cas d'erreur (défaut: 3)
- `--retry-delay`: Délai en secondes entre les tentatives (défaut: 5)

Options spécifiques pour OpenRouter:
- `--openrouter-key`: Clé API OpenRouter (ou définir la variable d'environnement OPENROUTER_API_KEY)
- `--max-tokens`: Nombre maximum de tokens à générer (défaut: 2048)

### Configuration et dépannage de LM Studio

#### Configuration:
1. Chargez un modèle puissant dans LM Studio (Llama3, Mixtral ou Gemma recommandés)
2. Activez le serveur local: Menu "Local Server" → "Enable"
3. Vérifiez le port utilisé (habituellement 1234) et ajustez avec `--lmstudio-port` si nécessaire

#### Résolution des problèmes de timeout:
1. **Augmentez le timeout**: `--timeout 300` (5 minutes)
2. **Réduisez la taille des chunks**: `--chunk-size 2000` (moins de texte à traiter à la fois)
3. **Utilisez un modèle plus performant**: Les modèles plus grands gèrent mieux les requêtes complexes
4. **Augmentez le nombre de retries**: `--max-retries 5` pour plus de tentatives automatiques

#### Messages d'erreur courants:
- **"Read timed out"**: Le modèle prend trop de temps pour répondre. Augmentez `--timeout` ou réduisez `--chunk-size`
- **"Erreur de décodage JSON"**: Le modèle n'a pas généré un JSON valide. Utilisez l'option `--retry` pour des requêtes simplifiées

![Configuration LM Studio](https://lmstudio.ai/docs/app/basics/presets)

### 7. Traitement avancé des données extraites

Nouvelle fonctionnalité ! Traiter les données extraites selon les préférences de l'utilisateur:

```bash
# Filtrer les articles des 30 derniers jours
python process_data.py articles.json --filter-date --days 30 -o articles_recents.json

# Analyser le sentiment des titres avec un modèle Hugging Face
python process_data.py articles.json --analyze-sentiment --sentiment-field titre -o articles_sentiment.json

# Catégoriser les articles et trier par sentiment
python process_data.py articles.json --categorize --analyze-sentiment --sort-by sentiment_score --sort-desc -o articles_classes.json
```

Options de traitement:
- **Filtrage par date**: `--filter-date`, `--date-field`, `--days`, `--start-date`, `--end-date`
- **Analyse de sentiment**: `--analyze-sentiment`, `--sentiment-field`, `--sentiment-model`, `--sentiment-provider`
- **Catégorisation**: `--categorize`, `--category-field`, `--categories`, `--category-model`, `--category-provider`
- **Tri et filtrage**: `--sort-by`, `--sort-desc`, `--filter`

Providers disponibles:
- `huggingface`: Modèles Hugging Face (locaux)
- `openai`: API OpenAI (GPT)
- `ollama`: Modèles locaux via Ollama

### 8. Export des données en DataFrame et CSV

Nouvelle fonctionnalité ! Organiser les données extraites dans un DataFrame pandas et les exporter au format CSV:

```bash
# Exporter des données structurées en CSV
python export_data.py articles.json -o articles.csv

# Nettoyer, filtrer et formater les données avant export
python export_data.py articles.json --no-duplicates --date-columns date --sort-by date --desc -o articles_tries.csv

# Sélectionner uniquement certaines colonnes
python export_data.py articles.json --columns titre date auteur categorie --preview -o selection.csv
```

Options d'export:
- **Formatage**: `--no-duplicates`, `--date-columns`, `--date-format`, `--sort-by`, `--desc`, `--columns`  
- **Options CSV**: `--delimiter`, `--encoding`, `--index`
- **Aperçu**: `--preview`, `--head`

## Exemples d'utilisation avancée

1. Pipeline complet: récupération, extraction, analyse et export:
```bash
# Étape 1: Récupérer et prétraiter le contenu
python web_scraper.py example.com --method selenium --main-content -o data/exemple.html

# Étape 2: Extraire des données structurées
python extract_data.py data/exemple.html "Extraire les titres, dates et auteurs des articles" -o data/articles.json

# Étape 3: Analyser et organiser les données
python process_data.py data/articles.json --filter-date --days 60 --analyze-sentiment -o data/articles_analyses.json

# Étape 4: Exporter les données en CSV pour utilisation dans d'autres outils
python export_data.py data/articles_analyses.json --sort-by sentiment_score --desc --preview -o data/articles_final.csv
```

2. Utilisation programmatique:
```python
from src.scrapers import fetch_content
from src.processors import extract_main_content
from src.llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results
from src.processors.data_processor import analyze_sentiment, filter_by_date, sort_and_filter
import pandas as pd

# Obtenir et traiter du contenu
html = fetch_content("https://example.com")
content = extract_main_content(html)

# Extraire des données structurées
llm = get_llm_provider("openai", model="gpt-3.5-turbo")
data = extract_data_from_chunks([content], "Extraire tous les titres et dates", llm)
articles = aggregate_extraction_results(data)

# Traiter les données
articles = filter_by_date(articles, days=30)
articles = analyze_sentiment(articles, text_field="titre")
articles = sort_and_filter(articles, sort_by="sentiment_score", ascending=False)

# Créer un DataFrame et exporter en CSV
df = pd.DataFrame(articles)
df.to_csv('articles_analyses.csv', index=False)
print(f"Données exportées dans {os.path.abspath('articles_analyses.csv')}")

# Afficher un aperçu
print(df.head())
```

## API REST avec FastAPI

Le projet dispose désormais d'une API REST complète qui expose toutes les fonctionnalités via des endpoints HTTP.

## Utilisation de l'API REST

### Installation des dépendances

Avant de lancer l'API, assurez-vous d'installer les dépendances nécessaires :

```bash
pip install fastapi uvicorn python-dotenv
```

### Lancer le serveur API

Pour démarrer le serveur API, utilisez le script `run_api.py` :

```bash
# Mode développement avec rechargement automatique du code
python run_api.py --reload

# Mode production avec plusieurs workers
python run_api.py --host 0.0.0.0 --port 8080 --workers 4
```

Options disponibles :
- `--host` : Adresse d'écoute (défaut : 127.0.0.1)
- `--port` : Port d'écoute (défaut : 8000)
- `--reload` : Activer le rechargement automatique du code (développement)
- `--workers` : Nombre de workers (défaut : 1)
- `--log-level` : Niveau de journalisation (défaut : info)

### Accéder à l'API

Une fois le serveur démarré, vous pouvez :

- Accéder à la documentation Swagger : http://localhost:8000/docs
- Vérifier l'état de l'API : http://localhost:8000/health
- Voir la page d'accueil : http://localhost:8000/

### Exemple de requête

```bash
# Exemple de requête avec curl pour récupérer une page web
curl -X POST http://localhost:8000/scraping/ \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "preprocess": true}'
```
