"""
Provider pour l'API OpenRouter permettant l'accès à de nombreux modèles de langage.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class OpenRouterProvider:
    """
    Provider pour l'API OpenRouter permettant d'accéder à divers modèles de langage.
    """
    
    def __init__(self, 
                api_key=None, 
                model="openai/gpt-3.5-turbo", 
                temperature=0.0,
                max_tokens=4096,
                timeout=120,
                **kwargs):
        """
        Initialise le provider OpenRouter.
        
        Args:
            api_key (str, optional): Clé API OpenRouter
            model (str): Modèle à utiliser (format: "provider/model")
            temperature (float): Température pour la génération (0.0-2.0)
            max_tokens (int): Nombre maximum de tokens à générer
            timeout (int): Délai d'attente pour les requêtes en secondes
            **kwargs: Arguments supplémentaires pour l'API
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("Aucune clé API OpenRouter fournie. Utilisez OPENROUTER_API_KEY ou passez api_key.")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_params = kwargs
        
        # URL de base de l'API OpenRouter
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def extract(self, content: str, instruction: str, output_format: str = "json") -> Dict[str, Any]:
        """
        Extrait des informations du contenu selon l'instruction via OpenRouter.
        
        Args:
            content (str): Contenu HTML/texte à analyser
            instruction (str): Instruction d'extraction
            output_format (str): Format de sortie souhaité (json, markdown, text)
            
        Returns:
            Dict[str, Any]: Résultat de l'extraction
        """
        if not self.api_key:
            raise ValueError("Clé API OpenRouter manquante. Définissez OPENROUTER_API_KEY ou passez api_key.")
        
        # Construire le système de messages
        if output_format.lower() == "json":
            system_prompt = (
                "Tu es un assistant spécialisé dans l'extraction de données à partir de contenu HTML. "
                "Analyse le contenu et extrait les informations demandées selon l'instruction. "
                "Réponds uniquement avec un objet JSON valide, sans texte avant ou après. "
                "N'utilise pas de bloc de code markdown. Commence directement par { et termine par }. "
                "Si tu ne trouves pas d'information, renvoie un objet avec des tableaux vides."
            )
        else:
            system_prompt = (
                f"Tu es un assistant spécialisé dans l'extraction de données à partir de contenu HTML. "
                f"Analyse le contenu et extrait les informations demandées selon l'instruction. "
                f"Réponds uniquement avec les données extraites au format {output_format}. "
                f"Si tu ne trouves pas d'information, renvoie un tableau ou une liste vide."
            )
        
        # Limiter la taille du contenu si nécessaire
        if len(content) > 25000:
            logger.warning("Le contenu a été tronqué car trop long (>25000 caractères)")
            content = content[:25000] + "[... contenu tronqué pour limite de taille ...]"
        
        user_prompt = f"### Instruction:\n{instruction}\n\n### Contenu HTML à analyser:\n{content}"
        
        # Préparer les données de la requête
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_params
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-scrapping-toolkit.com",  # Domaine de référence (peut être fictif)
            "X-Title": "AI Scrapping Toolkit"  # Titre de l'application
        }
        
        try:
            # Faire la requête API
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Extraire la réponse
            response_data = response.json()
            result = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Si format JSON demandé, parser la réponse
            if output_format.lower() == "json":
                try:
                    # Essayer d'extraire un bloc JSON s'il est entouré de ```
                    if result.startswith("```json") and result.endswith("```"):
                        result = result[7:-3].strip()
                    elif result.startswith("```") and result.endswith("```"):
                        result = result[3:-3].strip()
                    
                    # Trouver le premier { et le dernier } au cas où il y aurait du texte avant/après
                    first_brace = result.find("{")
                    last_brace = result.rfind("}")
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        result = result[first_brace:last_brace+1].strip()
                    
                    # Parser le JSON
                    return json.loads(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Erreur de décodage JSON: {str(e)}. Retour de la réponse brute.")
                    return {"raw_response": result, "error": "parsing_error"}
            
            return {"raw_response": result}
            
        except requests.exceptions.Timeout:
            logger.error(f"Délai d'attente dépassé lors de la requête à OpenRouter (timeout: {self.timeout}s)")
            return {"error": "timeout_error", "message": f"La requête a dépassé le délai de {self.timeout} secondes"}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de l'appel à l'API OpenRouter: {str(e)}")
            return {"error": str(e), "content": content[:100] + "..."}
            
        except Exception as e:
            logger.error(f"Erreur générale lors de l'interaction avec OpenRouter: {str(e)}")
            return {"error": str(e), "content": content[:100] + "..."}
