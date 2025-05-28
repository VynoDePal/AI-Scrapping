"""
Module pour le traitement avancé des données extraites (filtrage, tri, classification).
"""

import re
import json
import logging
import datetime
from typing import List, Dict, Any, Union, Optional, Callable
import pandas as pd

# Configuration du logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import conditionnel pour les dépendances optionnelles
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Hugging Face Transformers n'est pas disponible. Les fonctions d'analyse avancées seront limitées.")
    TRANSFORMERS_AVAILABLE = False

def convert_to_dataframe(data: Dict[str, List[Any]]) -> pd.DataFrame:
    """
    Convertit des données structurées en DataFrame pandas.
    
    Args:
        data (Dict[str, List[Any]]): Données extraites (ex: {'titres': [...], 'dates': [...]})
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    # S'il n'y a qu'une seule clé avec une liste, transformer en DataFrame directement
    if len(data) == 1:
        key = list(data.keys())[0]
        return pd.DataFrame(data[key])
    
    # Cas des données non structurées ou de structures différentes
    try:
        # Essayer d'abord de créer un DataFrame directement
        df = pd.DataFrame(data)
        return df
    except ValueError:
        # Si les listes ont des longueurs différentes, créer un DataFrame par clé
        dfs = {}
        for key, values in data.items():
            dfs[key] = pd.DataFrame(values)
        
        # Retourner le DataFrame le plus complet ou une concaténation
        largest_df_key = max(dfs, key=lambda k: len(dfs[k]))
        return dfs[largest_df_key]

def parse_date(date_str: str) -> Optional[datetime.datetime]:
    """
    Parse une date à partir d'une chaîne de caractères en détectant automatiquement le format.
    
    Args:
        date_str (str): Chaîne de caractères représentant une date
        
    Returns:
        datetime.datetime ou None: Date parsée ou None si impossible
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Nettoyage de la chaîne
    date_str = date_str.strip()
    
    # Formats de date courants (français et internationaux)
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',  # JJ/MM/AAAA
        '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d',  # AAAA/MM/JJ
        '%d/%m/%Y %H:%M', '%d-%m-%Y %H:%M',  # JJ/MM/AAAA HH:MM
        '%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M',  # AAAA/MM/JJ HH:MM
        '%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S',  # Avec secondes
        '%d %B %Y', '%d %b %Y',  # JJ Mois AAAA
        '%B %d, %Y', '%b %d, %Y'  # Mois JJ, AAAA
    ]
    
    # Essayer chaque format
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Essayer avec dateutil.parser si disponible
    try:
        from dateutil import parser
        return parser.parse(date_str, fuzzy=True)
    except (ImportError, ValueError):
        pass
    
    # Extraction de date avec regex si tout échoue
    patterns = [
        r'(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2,4})',  # JJ/MM/AAAA ou AAAA/MM/JJ
        r'(\d{4})[/\.-](\d{1,2})[/\.-](\d{1,2})'     # AAAA/MM/JJ
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            try:
                if len(groups[0]) == 4:  # Format AAAA/MM/JJ
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                else:  # Format JJ/MM/AAAA
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    # Corriger l'année si nécessaire
                    if year < 100:
                        year += 2000 if year < 50 else 1900
                
                return datetime.datetime(year, month, day)
            except ValueError:
                continue
    
    return None

def filter_by_date(
    data: Dict[str, List[Any]], 
    date_field: str = 'date', 
    days: int = 30, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, List[Any]]:
    """
    Filtre les données par date.
    
    Args:
        data (Dict): Données extraites
        date_field (str): Nom du champ contenant la date
        days (int): Nombre de jours à considérer (par défaut: 30 derniers jours)
        start_date (str, optional): Date de début au format YYYY-MM-DD
        end_date (str, optional): Date de fin au format YYYY-MM-DD
        
    Returns:
        Dict: Données filtrées
    """
    # Vérifie si les données contiennent le champ de date
    date_key = None
    for key in data.keys():
        if date_field.lower() in key.lower():
            date_key = key
            break
    
    # Si aucun champ de date trouvé ou si le champ est vide
    if not date_key or not data[date_key]:
        logger.warning(f"Aucun champ de date correspondant à '{date_field}' trouvé dans les données")
        return data
    
    # Convertir en DataFrame pour faciliter le filtrage
    df = convert_to_dataframe(data)
    
    # Déterminer la colonne de date
    date_columns = [col for col in df.columns if date_field.lower() in col.lower()]
    if not date_columns:
        logger.warning(f"Aucune colonne de date trouvée dans le DataFrame")
        return data
    
    date_column = date_columns[0]
    
    # Convertir la colonne de date en datetime
    df['parsed_date'] = df[date_column].apply(parse_date)
    
    # Filtrer les entrées sans date valide
    df_valid = df.dropna(subset=['parsed_date'])
    if len(df_valid) == 0:
        logger.warning("Aucune date valide n'a pu être parsée")
        return data
    
    # Définir les dates de début et de fin pour le filtrage
    if start_date:
        start_dt = parse_date(start_date)
    else:
        start_dt = datetime.datetime.now() - datetime.timedelta(days=days)
    
    if end_date:
        end_dt = parse_date(end_date)
    else:
        end_dt = datetime.datetime.now()
    
    # Filtrer par date
    filtered_df = df_valid[(df_valid['parsed_date'] >= start_dt) & 
                           (df_valid['parsed_date'] <= end_dt)]
    
    # Convertir le résultat filtré en dictionnaire
    filtered_data = {}
    for key in data.keys():
        if key in filtered_df.columns:
            filtered_data[key] = filtered_df[key].tolist()
    
    # Ajouter les dates parsées si demandé
    filtered_data['dates_parsées'] = filtered_df['parsed_date'].dt.strftime('%Y-%m-%d').tolist()
    
    return filtered_data

def analyze_sentiment(
    data: Dict[str, List[Any]], 
    text_field: str = 'titre',
    model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment',
    provider: str = 'huggingface'
) -> Dict[str, List[Any]]:
    """
    Analyse le sentiment des textes dans les données.
    
    Args:
        data (Dict): Données extraites
        text_field (str): Nom du champ contenant le texte à analyser
        model_name (str): Nom du modèle à utiliser
        provider (str): Provider du modèle (huggingface, openai, ollama)
        
    Returns:
        Dict: Données avec scores de sentiment ajoutés
    """
    # Vérifie si les données contiennent le champ de texte
    text_key = None
    for key in data.keys():
        if text_field.lower() in key.lower():
            text_key = key
            break
    
    # Si aucun champ de texte trouvé ou si le champ est vide
    if not text_key or not data[text_key]:
        logger.warning(f"Aucun champ de texte correspondant à '{text_field}' trouvé dans les données")
        return data
    
    # Préparer les données
    texts = data[text_key]
    
    # Créer une copie des données pour ne pas modifier l'original
    result_data = {k: v.copy() if isinstance(v, list) else v for k, v in data.items()}
    
    # Analyser le sentiment selon le provider choisi
    if provider.lower() == 'huggingface':
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Hugging Face Transformers n'est pas disponible. Impossible d'analyser le sentiment.")
            result_data['sentiment_score'] = [0] * len(texts)
            result_data['sentiment'] = ['neutre'] * len(texts)
            return result_data
        
        # Utiliser Hugging Face pour l'analyse de sentiment
        try:
            sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)
            
            # Analyser par lots pour éviter les problèmes de mémoire
            batch_size = 8
            sentiments = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = sentiment_analyzer(batch)
                sentiments.extend(batch_results)
            
            # Extraire les scores et les labels
            scores = [item['score'] for item in sentiments]
            labels = [item['label'] for item in sentiments]
            
            # Ajouter les résultats
            result_data['sentiment_score'] = scores
            result_data['sentiment'] = labels
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment avec Hugging Face: {str(e)}")
            result_data['sentiment_score'] = [0] * len(texts)
            result_data['sentiment'] = ['erreur'] * len(texts)
    
    elif provider.lower() == 'openai':
        # Utiliser OpenAI pour l'analyse de sentiment
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                logger.error("Clé API OpenAI non trouvée dans les variables d'environnement")
                result_data['sentiment_score'] = [0] * len(texts)
                result_data['sentiment'] = ['erreur'] * len(texts)
                return result_data
            
            openai.api_key = api_key
            
            scores = []
            labels = []
            
            for text in texts:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Analyse le sentiment du texte suivant et réponds uniquement avec un des labels suivants: positif, neutre, négatif."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0
                )
                
                sentiment = response.choices[0].message.content.strip().lower()
                
                # Convertir en score numérique
                if sentiment == "positif":
                    score = 0.9
                elif sentiment == "neutre":
                    score = 0.5
                else:
                    score = 0.1
                
                scores.append(score)
                labels.append(sentiment)
            
            # Ajouter les résultats
            result_data['sentiment_score'] = scores
            result_data['sentiment'] = labels
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment avec OpenAI: {str(e)}")
            result_data['sentiment_score'] = [0] * len(texts)
            result_data['sentiment'] = ['erreur'] * len(texts)
    
    elif provider.lower() == 'ollama':
        # Utiliser Ollama pour l'analyse de sentiment
        try:
            import requests
            
            scores = []
            labels = []
            
            for text in texts:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        "model": "llama2",
                        "prompt": f"Analyse le sentiment du texte suivant et réponds uniquement avec un des labels suivants: positif, neutre, négatif.\nTexte: {text}",
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    sentiment = response.json().get('response', '').strip().lower()
                    
                    # Convertir en score numérique
                    if "positif" in sentiment:
                        score = 0.9
                        label = "positif"
                    elif "neutre" in sentiment:
                        score = 0.5
                        label = "neutre"
                    else:
                        score = 0.1
                        label = "négatif"
                    
                    scores.append(score)
                    labels.append(label)
                else:
                    scores.append(0)
                    labels.append("erreur")
            
            # Ajouter les résultats
            result_data['sentiment_score'] = scores
            result_data['sentiment'] = labels
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment avec Ollama: {str(e)}")
            result_data['sentiment_score'] = [0] * len(texts)
            result_data['sentiment'] = ['erreur'] * len(texts)
    
    else:
        logger.warning(f"Provider '{provider}' non reconnu pour l'analyse de sentiment")
        result_data['sentiment_score'] = [0] * len(texts)
        result_data['sentiment'] = ['non analysé'] * len(texts)
    
    return result_data

def categorize_text(
    data: Dict[str, List[Any]], 
    text_field: str = 'titre',
    categories: List[str] = None,
    model_name: str = 'facebook/bart-large-mnli',
    provider: str = 'huggingface'
) -> Dict[str, List[Any]]:
    """
    Catégorise les textes selon des catégories prédéfinies.
    
    Args:
        data (Dict): Données extraites
        text_field (str): Nom du champ contenant le texte à catégoriser
        categories (List[str]): Liste des catégories
        model_name (str): Nom du modèle à utiliser
        provider (str): Provider du modèle (huggingface, openai, ollama)
        
    Returns:
        Dict: Données avec catégories ajoutées
    """
    # Catégories par défaut si non spécifiées
    if categories is None:
        categories = [
            "Technologie", "Science", "Politique", "Économie", 
            "Santé", "Éducation", "Environnement", "Sport", 
            "Culture", "Divertissement"
        ]
    
    # Vérifie si les données contiennent le champ de texte
    text_key = None
    for key in data.keys():
        if text_field.lower() in key.lower():
            text_key = key
            break
    
    # Si aucun champ de texte trouvé ou si le champ est vide
    if not text_key or not data[text_key]:
        logger.warning(f"Aucun champ de texte correspondant à '{text_field}' trouvé dans les données")
        return data
    
    # Préparer les données
    texts = data[text_key]
    
    # Créer une copie des données pour ne pas modifier l'original
    result_data = {k: v.copy() if isinstance(v, list) else v for k, v in data.items()}
    
    # Catégoriser selon le provider choisi
    if provider.lower() == 'huggingface':
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Hugging Face Transformers n'est pas disponible. Impossible de catégoriser.")
            result_data['catégorie'] = ['Non classé'] * len(texts)
            return result_data
        
        # Utiliser Hugging Face pour la classification zéro-shot
        try:
            classifier = pipeline("zero-shot-classification", model=model_name)
            
            # Catégoriser par lots pour éviter les problèmes de mémoire
            batch_size = 8
            all_results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = []
                
                for text in batch:
                    result = classifier(text, categories)
                    top_category = result['labels'][0]
                    top_score = result['scores'][0]
                    batch_results.append({'category': top_category, 'score': top_score})
                
                all_results.extend(batch_results)
            
            # Extraire les catégories et les scores
            result_categories = [item['category'] for item in all_results]
            category_scores = [item['score'] for item in all_results]
            
            # Ajouter les résultats
            result_data['catégorie'] = result_categories
            result_data['catégorie_score'] = category_scores
            
        except Exception as e:
            logger.error(f"Erreur lors de la catégorisation avec Hugging Face: {str(e)}")
            result_data['catégorie'] = ['Erreur'] * len(texts)
    
    elif provider.lower() in ['openai', 'ollama']:
        # Code pour OpenAI ou Ollama
        try:
            if provider.lower() == 'openai':
                from src.llm.providers.openai_provider import OpenAIProvider
                llm = OpenAIProvider(temperature=0.0)
            else:
                from src.llm.providers.ollama_provider import OllamaProvider
                llm = OllamaProvider(temperature=0.0)
            
            categories_str = ", ".join(categories)
            result_categories = []
            
            for text in texts:
                instruction = f"Catégorise le texte suivant dans une seule de ces catégories: {categories_str}. Réponds uniquement avec le nom de la catégorie."
                
                try:
                    result = llm.extract(text, instruction, output_format="text")
                    category = result.get('raw_response', 'Non classé').strip()
                    
                    # Vérifier si la catégorie est dans la liste
                    if category not in categories:
                        # Trouver la catégorie la plus proche
                        for cat in categories:
                            if cat.lower() in category.lower():
                                category = cat
                                break
                        else:
                            category = 'Autre'
                    
                    result_categories.append(category)
                except Exception as e:
                    logger.error(f"Erreur lors de la catégorisation d'un texte: {str(e)}")
                    result_categories.append('Erreur')
            
            # Ajouter les résultats
            result_data['catégorie'] = result_categories
            
        except Exception as e:
            logger.error(f"Erreur lors de la catégorisation avec {provider}: {str(e)}")
            result_data['catégorie'] = ['Erreur'] * len(texts)
    
    else:
        logger.warning(f"Provider '{provider}' non reconnu pour la catégorisation")
        result_data['catégorie'] = ['Non classé'] * len(texts)
    
    return result_data

def sort_and_filter(
    data: Dict[str, List[Any]],
    sort_by: Optional[str] = None,
    ascending: bool = True,
    filter_expr: Optional[str] = None
) -> Dict[str, List[Any]]:
    """
    Trie et filtre les données selon des critères spécifiés.
    
    Args:
        data (Dict): Données extraites
        sort_by (str, optional): Champ pour le tri
        ascending (bool): Ordre croissant ou décroissant
        filter_expr (str, optional): Expression de filtrage (ex: "sentiment == 'positif'")
        
    Returns:
        Dict: Données triées et filtrées
    """
    # Convertir en DataFrame pour faciliter le tri et le filtrage
    df = convert_to_dataframe(data)
    
    # Appliquer le filtrage si une expression est fournie
    if filter_expr:
        try:
            filtered_df = df.query(filter_expr)
            if len(filtered_df) == 0:
                logger.warning(f"Le filtre '{filter_expr}' a éliminé toutes les données")
                return data
            df = filtered_df
        except Exception as e:
            logger.error(f"Erreur lors du filtrage avec l'expression '{filter_expr}': {str(e)}")
    
    # Appliquer le tri si un champ est spécifié
    if sort_by:
        # Vérifier si le champ de tri existe
        matching_cols = [col for col in df.columns if sort_by.lower() in col.lower()]
        if matching_cols:
            sort_column = matching_cols[0]
            try:
                df = df.sort_values(by=sort_column, ascending=ascending)
            except Exception as e:
                logger.error(f"Erreur lors du tri par '{sort_column}': {str(e)}")
        else:
            logger.warning(f"Champ de tri '{sort_by}' non trouvé dans les données")
    
    # Convertir le DataFrame en dictionnaire
    result_data = {}
    for column in df.columns:
        result_data[column] = df[column].tolist()
    
    return result_data

def process_data(
    data: Dict[str, List[Any]],
    operations: List[Dict[str, Any]]
) -> Dict[str, List[Any]]:
    """
    Traite les données selon une liste d'opérations.
    
    Args:
        data (Dict): Données extraites
        operations (List[Dict]): Liste d'opérations à appliquer
            [{"type": "filter_by_date", "params": {...}}, ...]
        
    Returns:
        Dict: Données traitées
    """
    result = data
    
    # Appliquer chaque opération séquentiellement
    for operation in operations:
        op_type = operation.get('type')
        params = operation.get('params', {})
        
        if op_type == 'filter_by_date':
            result = filter_by_date(result, **params)
        elif op_type == 'analyze_sentiment':
            result = analyze_sentiment(result, **params)
        elif op_type == 'categorize_text':
            result = categorize_text(result, **params)
        elif op_type == 'sort_and_filter':
            result = sort_and_filter(result, **params)
        else:
            logger.warning(f"Type d'opération non reconnu: {op_type}")
    
    return result
