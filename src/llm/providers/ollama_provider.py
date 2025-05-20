"""
Provider pour l'API Ollama (modèles locaux).
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class OllamaProvider:
    """
    Provider pour l'API Ollama (modèles locaux).
    """
    
    def __init__(self, host="http://localhost:11434", model="llama2", temperature=0.0, **kwargs):
        """
        Initialise le provider Ollama.
        
        Args:
            host (str): URL de l'API Ollama
            model (str): Modèle à utiliser (llama2, mistral, etc.)
            temperature (float): Température pour la génération
            **kwargs: Arguments supplémentaires pour l'API
        """
        self.host = host
        self.model = model
        self.temperature = temperature
        self.extra_params = kwargs
    
    def extract(self, content: str, instruction: str, output_format: str = "json") -> Dict[str, Any]:
        """
        Extrait des informations du contenu selon l'instruction via Ollama.
        
        Args:
            content (str): Contenu HTML/texte à analyser
            instruction (str): Instruction d'extraction
            output_format (str): Format de sortie souhaité (json, markdown, text)
            
        Returns:
            Dict[str, Any]: Résultat de l'extraction
        """
        # Construire le prompt système + utilisateur
        system_prompt = (
            f"Tu es un assistant spécialisé dans l'extraction de données à partir de contenu HTML. "
            f"Analyse le contenu et extrait les informations demandées selon l'instruction. "
            f"Réponds uniquement avec les données extraites au format {output_format}. "
            f"Si tu ne trouves pas d'information, renvoie un tableau/objet vide."
        )
        
        # Construire le prompt complet
        prompt = f"{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Contenu:\n{content}"
        
        try:
            # Requête à l'API Ollama
            url = f"{self.host}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False,
                **self.extra_params
            }
            
            # Envoyer la requête
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            # Extraire la réponse
            result = response.json().get("response", "").strip()
            
            # Si format JSON demandé, parser la réponse
            if output_format.lower() == "json":
                try:
                    # Essayer d'extraire un bloc JSON s'il est entouré de ```
                    if result.startswith("```json") and result.endswith("```"):
                        result = result[7:-3].strip()
                    elif result.startswith("```") and result.endswith("```"):
                        result = result[3:-3].strip()
                    
                    # Parser le JSON
                    return json.loads(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Erreur de décodage JSON: {str(e)}. Retour de la réponse brute.")
                    return {"raw_response": result, "error": "parsing_error"}
            
            return {"raw_response": result}
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API Ollama: {str(e)}")
            return {"error": str(e), "content": content[:100] + "..."}
