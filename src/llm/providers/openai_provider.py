"""
Provider pour l'API OpenAI.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class OpenAIProvider:
    """
    Provider pour l'API OpenAI.
    """
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo", temperature=0.0, **kwargs):
        """
        Initialise le provider OpenAI.
        
        Args:
            api_key (str, optional): Clé API OpenAI
            model (str): Modèle à utiliser (gpt-3.5-turbo, gpt-4, etc.)
            temperature (float): Température pour la génération (0.0-2.0)
            **kwargs: Arguments supplémentaires pour l'API
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("Aucune clé API OpenAI fournie. Utilisez OPENAI_API_KEY ou passez api_key.")
        
        self.model = model
        self.temperature = temperature
        self.extra_params = kwargs
        
        # Tenter d'importer la bibliothèque OpenAI
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
        except ImportError:
            logger.error("La bibliothèque OpenAI n'est pas installée. Exécutez 'pip install openai'.")
            raise
    
    def extract(self, content: str, instruction: str, output_format: str = "json") -> Dict[str, Any]:
        """
        Extrait des informations du contenu selon l'instruction.
        
        Args:
            content (str): Contenu HTML/texte à analyser
            instruction (str): Instruction d'extraction
            output_format (str): Format de sortie souhaité (json, markdown, text)
            
        Returns:
            Dict[str, Any]: Résultat de l'extraction
        """
        if not self.api_key:
            raise ValueError("Clé API OpenAI manquante. Définissez OPENAI_API_KEY ou passez api_key.")
        
        # Construire le système de messages
        system_prompt = (
            f"Tu es un assistant spécialisé dans l'extraction de données à partir de contenu HTML. "
            f"Analyse le contenu et extrait les informations demandées selon l'instruction. "
            f"Réponds uniquement avec les données extraites au format {output_format}. "
            f"Si tu ne trouves pas d'information, renvoie un tableau/objet vide."
        )
        
        user_prompt = f"### Instruction:\n{instruction}\n\n### Contenu:\n{content}"
        
        try:
            # Faire la requête API
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                **self.extra_params
            )
            
            # Extraire la réponse
            result = response.choices[0].message.content.strip()
            
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
            logger.error(f"Erreur lors de l'appel à l'API OpenAI: {str(e)}")
            return {"error": str(e), "content": content[:100] + "..."}
