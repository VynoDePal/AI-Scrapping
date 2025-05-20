"""
Module d'analyse de requêtes en langage naturel pour extraire 
les intentions et entités, puis rechercher des chunks pertinents.
"""

import re
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

# Import des modules de NLP
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
    import torch
except ImportError:
    logging.warning("Transformers non disponible. L'analyse avancée sera limitée.")

# Import des modules d'embeddings
from ..embeddings import chunks_to_embeddings, search_similar
from sentence_transformers import SentenceTransformer

# Configuration du logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Catégories d'intentions reconnues
INTENT_CATEGORIES = {
    "extraction": ["extraire", "trouver", "chercher", "obtenir", "sortir", "récupérer", "collecter", "identifier"],
    "analyse": ["analyser", "évaluer", "comparer", "étudier", "examiner"],
    "résumé": ["résumer", "synthétiser", "condenser", "raccourcir"],
    "traduction": ["traduire", "convertir", "transformer"]
}

# Patterns regex pour la détection rapide d'entités
ENTITY_PATTERNS = {
    "date": r"\b(date|jour|mois|année)\b|\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b",
    "titre": r"\b(titre|heading|header|h1|h2|h3)\b",
    "prix": r"\b(prix|tarif|coût|cout|euro|dollar|€|\$)\b",
    "auteur": r"\b(auteur|écrivain|rédacteur|journaliste)\b",
    "image": r"\b(image|photo|illustration|figure)\b"
}

def load_nlp_model(task: str = "ner", model_name: Optional[str] = None):
    """
    Charge un modèle NLP pour une tâche spécifique.
    
    Args:
        task (str): Tâche NLP ('ner', 'intent', 'qa')
        model_name (str, optional): Nom du modèle spécifique à charger
        
    Returns:
        Le modèle chargé ou None en cas d'erreur
    """
    try:
        if task == "ner":
            # Modèle de reconnaissance d'entités nommées
            if not model_name:
                model_name = "Jean-Baptiste/camembert-ner"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            return pipeline("ner", model=model, tokenizer=tokenizer)
        
        elif task == "intent":
            # Modèle de classification d'intentions
            if not model_name:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                
            return pipeline("text-classification", model=model_name)
        
        elif task == "qa":
            # Modèle de question-réponse
            if not model_name:
                model_name = "etalab-ia/camembert-base-squadFR-fquad-piaf"
                
            return pipeline("question-answering", model=model_name)
        
        else:
            logger.error(f"Tâche '{task}' non reconnue.")
            return None
            
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle '{task}': {str(e)}")
        return None

def extract_entities_and_intent(query: str, 
                                use_transformers: bool = False,
                                nlp_model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Extrait les entités et l'intention d'une requête utilisateur.
    
    Args:
        query (str): Requête utilisateur en langage naturel
        use_transformers (bool): Utiliser les modèles Hugging Face pour une analyse plus précise
        nlp_model (Any, optional): Modèle NLP préchargé
        
    Returns:
        Dict: Dictionnaire contenant les entités et l'intention extraites
    """
    # Conversion en minuscule pour l'analyse
    query_lower = query.lower()
    result = {
        "query": query,
        "entities": [],
        "intent": "unknown",
        "intent_confidence": 0.0
    }
    
    # 1. Analyse basique avec regex pour les entités
    for entity_type, pattern in ENTITY_PATTERNS.items():
        if re.search(pattern, query_lower):
            result["entities"].append({
                "type": entity_type,
                "value": re.search(pattern, query_lower).group(),
                "confidence": 0.7
            })
    
    # 2. Détection de l'intention avec mots-clés
    for intent, keywords in INTENT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in query_lower:
                result["intent"] = intent
                result["intent_confidence"] = 0.8
                break
                
    # 3. Si demandé, analyse avancée avec transformers
    if use_transformers and nlp_model is None:
        try:
            # Reconnaissance d'entités nommées
            ner_model = load_nlp_model("ner")
            if ner_model:
                entities = ner_model(query)
                
                # Regrouper les entités par type
                grouped_entities = {}
                for entity in entities:
                    entity_type = entity['entity']
                    if entity_type not in grouped_entities:
                        grouped_entities[entity_type] = {
                            "text": entity['word'],
                            "confidence": entity['score']
                        }
                    else:
                        # Si l'entité existe déjà, prendre celle avec le meilleur score
                        if entity['score'] > grouped_entities[entity_type]["confidence"]:
                            grouped_entities[entity_type] = {
                                "text": entity['word'],
                                "confidence": entity['score']
                            }
                
                # Ajouter les entités trouvées au résultat
                for entity_type, entity_data in grouped_entities.items():
                    result["entities"].append({
                        "type": entity_type,
                        "value": entity_data["text"],
                        "confidence": entity_data["confidence"]
                    })
                    
            # Classification d'intention
            intent_model = load_nlp_model("intent")
            if intent_model:
                intent = intent_model(query)
                result["intent"] = intent[0]['label']
                result["intent_confidence"] = intent[0]['score']
                
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse avec transformers: {str(e)}")
    
    return result

def analyze_query(query: str, 
                  use_transformers: bool = False,
                  nlp_model: Optional[Any] = None,
                  embedding_model: Optional[SentenceTransformer] = None) -> Dict[str, Any]:
    """
    Analyse complète d'une requête utilisateur.
    
    Args:
        query (str): Requête de l'utilisateur
        use_transformers (bool): Utiliser les modèles HF pour une analyse avancée
        nlp_model: Modèle de NLP préchargé
        embedding_model: Modèle d'embedding préchargé
        
    Returns:
        Dict: Résultat de l'analyse avec intention, entités et embedding
    """
    # 1. Extraction des entités et de l'intention
    analysis = extract_entities_and_intent(query, use_transformers, nlp_model)
    
    # 2. Génération de l'embedding de la requête
    if embedding_model:
        model = embedding_model
    else:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle d'embedding: {str(e)}")
            analysis["embedding"] = None
            return analysis
    
    try:
        embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
        analysis["embedding"] = embedding
        analysis["embedding_model"] = model.get_sentence_embedding_dimension()
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'embedding: {str(e)}")
        analysis["embedding"] = None
    
    return analysis

def search_with_query(query: str,
                     index,
                     index_metadata: Dict[str, Any],
                     model_name: str = 'all-MiniLM-L6-v2',
                     top_k: int = 5,
                     use_transformers: bool = False,
                     filter_by_entities: bool = True) -> List[Dict[str, Any]]:
    """
    Recherche des chunks en fonction d'une requête analysée.
    
    Args:
        query (str): Requête utilisateur en langage naturel
        index: Index FAISS
        index_metadata (Dict): Métadonnées de l'index
        model_name (str): Nom du modèle d'embedding
        top_k (int): Nombre de résultats à retourner
        use_transformers (bool): Utiliser transformers pour l'analyse
        filter_by_entities (bool): Filtrer les résultats selon les entités détectées
        
    Returns:
        List[Dict]: Liste des chunks les plus pertinents avec scores et métadonnées
    """
    # 1. Analyser la requête pour comprendre l'intention et les entités
    query_analysis = analyze_query(query, use_transformers)
    logger.info(f"Analyse de la requête: {json.dumps(query_analysis, indent=2, ensure_ascii=False)}")
    
    # 2. Chercher les chunks les plus similaires sémantiquement
    from ..embeddings.vector_db import search_similar
    
    results = search_similar(
        query=query, 
        index=index, 
        index_metadata=index_metadata,
        model_name=model_name,
        top_k=top_k * 2  # On récupère plus de résultats pour le filtrage
    )
    
    # 3. Filtrer et trier les résultats en fonction des entités si demandé
    if filter_by_entities and query_analysis["entities"]:
        filtered_results = []
        entity_keywords = [entity["value"] for entity in query_analysis["entities"]]
        
        for result in results:
            # Score bonus pour les chunks qui contiennent les entités recherchées
            entity_bonus = 0.0
            for keyword in entity_keywords:
                if keyword.lower() in result["chunk"].lower():
                    entity_bonus += 0.1  # Bonus pour chaque entité trouvée
            
            # Ajuster le score avec le bonus d'entité
            result["original_score"] = result["score"]
            result["score"] += entity_bonus
            result["contains_entities"] = entity_bonus > 0
            
            filtered_results.append(result)
        
        # Trier à nouveau avec les scores ajustés
        filtered_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limiter au nombre demandé
        results = filtered_results[:top_k]
    else:
        # Limiter au nombre demandé
        results = results[:top_k]
    
    # 4. Ajouter l'analyse de la requête aux résultats
    for result in results:
        result["query_analysis"] = query_analysis
    
    return results
