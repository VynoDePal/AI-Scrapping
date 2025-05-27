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
            error_msg = "Clé API OpenAI manquante. Définissez OPENAI_API_KEY ou passez api_key."
            logger.error(error_msg)
            raise ValueError(error_msg)

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
                f"Si tu ne trouves pas d'information, renvoie un tableau/objet vide."
            )

        # Limiter la taille du contenu si nécessaire
        if len(content) > 25000:
            logger.warning("Le contenu a été tronqué car trop long (>25000 caractères)")
            content = content[:25000] + "[... contenu tronqué pour limite de taille ...]"

        user_prompt = f"### Instruction:\n{instruction}\n\n### Contenu HTML à analyser:\n{content}"

        try:
            # Faire la requête API
            logger.debug(f"Envoi de la requête à l'API OpenAI (modèle: {self.model}, température: {self.temperature})")
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
            logger.debug(f"Réponse reçue de l'API OpenAI ({len(result)} caractères)")

            # Si format JSON demandé, parser la réponse
            if output_format.lower() == "json":
                try:
                    # Essayer d'extraire un bloc JSON s'il est entouré de ```
                    if result.startswith("```json") and result.endswith("```"):
                        result = result[7:-3].strip()
                        logger.debug("Bloc de code JSON détecté et extrait")
                    elif result.startswith("```") and result.endswith("```"):
                        result = result[3:-3].strip()
                        logger.debug("Bloc de code générique détecté et extrait")

                    # Trouver le premier { et le dernier } au cas où il y aurait du texte avant/après
                    first_brace = result.find("{")
                    last_brace = result.rfind("}")
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        result = result[first_brace:last_brace+1].strip()
                        logger.debug("JSON extrait des accolades")

                    # Parser le JSON
                    parsed_result = json.loads(result)
                    logger.debug("JSON parsé avec succès")
                    return parsed_result
                except json.JSONDecodeError as e:
                    error_msg = f"Erreur de décodage JSON: {str(e)}. Réponse brute: {result[:200]}..."
                    logger.warning(error_msg)
                    return {"raw_response": result, "error": "parsing_error", "message": str(e)}

            return {"raw_response": result}

        except self.openai.error.InvalidRequestError as e:
            error_msg = f"Requête invalide à l'API OpenAI: {str(e)}"
            logger.error(error_msg)
            return {"error": "invalid_request", "message": str(e), "content_length": len(content)}

        except self.openai.error.AuthenticationError as e:
            error_msg = f"Erreur d'authentification OpenAI: {str(e)}"
            logger.error(error_msg)
            return {"error": "authentication_error", "message": str(e)}

        except self.openai.error.RateLimitError as e:
            error_msg = f"Limite de débit OpenAI atteinte: {str(e)}"
            logger.error(error_msg)
            return {"error": "rate_limit", "message": str(e)}

        except self.openai.error.ServiceUnavailableError as e:
            error_msg = f"Service OpenAI indisponible: {str(e)}"
            logger.error(error_msg)
            return {"error": "service_unavailable", "message": str(e)}

        except self.openai.error.Timeout as e:
            error_msg = f"Délai d'attente dépassé pour l'API OpenAI: {str(e)}"
            logger.error(error_msg)
            return {"error": "timeout", "message": str(e)}

        except Exception as e:
            error_msg = f"Erreur lors de l'appel à l'API OpenAI: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Détails de l'erreur: {traceback.format_exc()}")
            return {"error": "api_error", "message": str(e), "content_preview": content[:100] + "..."}
