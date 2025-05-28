"""
Provider pour l'API LM Studio (modèles locaux).
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class LMStudioProvider:
    """
    Provider pour l'API LM Studio (serveur local).
    """
    
    def __init__(self, 
                host="http://localhost:1234/v1", 
                model=None, 
                temperature=0.0,
                timeout=180,  # Timeout augmenté à 3 minutes par défaut
                max_retries=3,
                retry_delay=5,
                **kwargs):
        """
        Initialise le provider LM Studio.
        
        Args:
            host (str): URL de l'API LM Studio
            model (str, optional): Nom du modèle à utiliser (ignoré par LM Studio qui utilise le modèle chargé)
            temperature (float): Température pour la génération
            timeout (int): Délai d'expiration en secondes pour les requêtes API
            max_retries (int): Nombre maximum de tentatives en cas d'erreur
            retry_delay (int): Délai en secondes entre les tentatives
            **kwargs: Arguments supplémentaires pour l'API
        """
        self.host = host
        self.model = model  # LM Studio utilise le modèle chargé dans l'interface, ce paramètre est ignoré
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_params = kwargs
        
        # Vérifier si LM Studio est accessible
        try:
            health_url = f"{self.host.split('/v1')[0]}/health"
            logger.info(f"Vérification de la disponibilité de LM Studio à {health_url}")
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ LM Studio est accessible à l'adresse {self.host}")
                # Vérifier si un modèle est bien chargé
                try:
                    models_url = f"{self.host.rstrip('/')}/models"
                    models_response = requests.get(models_url, timeout=5)
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        if models_data.get("data"):
                            logger.info(f"Modèle(s) disponible(s) dans LM Studio: {[m.get('id') for m in models_data.get('data', [])]}")
                        else:
                            logger.warning("Aucun modèle n'est actuellement chargé dans LM Studio")
                except Exception:
                    pass
            else:
                logger.warning(f"⚠️ LM Studio a répondu avec le code {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Impossible de contacter LM Studio à {self.host} - Le serveur est-il démarré?")
            logger.warning("Vérifiez que LM Studio est en cours d'exécution et que son serveur API est activé")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la vérification de LM Studio: {e}")
            logger.warning("Instructions: 1) Ouvrez LM Studio, 2) Cliquez sur 'Local Server', 3) Activez le serveur, 4) Vérifiez l'URL")
    
    def extract(self, content: str, instruction: str, output_format: str = "json") -> Dict[str, Any]:
        """
        Extrait des informations du contenu selon l'instruction via LM Studio.
        
        Args:
            content (str): Contenu HTML/texte à analyser
            instruction (str): Instruction d'extraction
            output_format (str): Format de sortie souhaité (json, markdown, text)
            
        Returns:
            Dict[str, Any]: Résultat de l'extraction
        """
        # Construire des prompts spécifiques selon le format demandé
        if output_format.lower() == "json":
            system_prompt = (
                "Tu es un assistant expert en extraction de données structurées. "
                "Tu dois analyser le contenu HTML fourni et extraire les informations selon l'instruction. "
                "INSTRUCTION TRÈS IMPORTANTE: "
                "1. Tu dois UNIQUEMENT répondre avec un objet JSON valide, sans texte avant ou après. "
                "2. N'utilise PAS de bloc de code markdown comme ```json ou ```. "
                "3. Commence directement par { et termine par }. "
                "4. Si tu ne trouves pas d'information, renvoie un objet avec des tableaux vides, exemple: "
                '{"titres": [], "dates": []}. '
                "5. Assure-toi que chaque valeur extraite est du bon type (string, number, etc.). "
                "6. Vérifie que ton JSON est correctement formaté avec des virgules entre les éléments."
            )
        else:
            system_prompt = (
                f"Tu es un assistant spécialisé dans l'extraction de données à partir de contenu HTML. "
                f"Analyse le contenu et extrait les informations demandées selon l'instruction. "
                f"Réponds uniquement avec les données extraites au format {output_format}. "
                f"Si tu ne trouves pas d'information, renvoie un tableau ou une liste vide."
            )
        
        # Limiter la taille du contenu si nécessaire pour éviter des problèmes de contexte
        if len(content) > 20000:
            logger.warning("Le contenu a été tronqué car trop long (>20000 caractères)")
            content = content[:20000] + "[... contenu tronqué pour limite de taille ...]"
        
        # Construire le prompt complet pour l'utilisateur
        user_prompt = f"### Instruction:\n{instruction}\n\n### Contenu HTML à analyser:\n{content}"
        
        # Préparer les données de la requête de base
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "stream": False,
            "max_tokens": 2048  # Limiter la taille de la réponse
        }
        
        # Ajouter les paramètres supplémentaires
        for key, value in self.extra_params.items():
            if key not in payload:
                payload[key] = value
        
        # Si model est spécifié, l'ajouter au payload
        if self.model:
            payload["model"] = self.model
        
        # Configuration de l'URL complète
        api_url = f"{self.host.rstrip('/')}/chat/completions"
        
        # Système de tentatives multiples
        retries = 0
        while retries <= self.max_retries:
            try:
                # Log de l'envoi de la requête
                if retries == 0:
                    logger.info(f"Envoi de la requête d'extraction à LM Studio: {api_url}")
                else:
                    logger.info(f"Tentative {retries}/{self.max_retries} d'envoi à LM Studio")
                
                # Ajuster la température en fonction du nombre de tentatives
                if retries > 0:
                    # Réduire la température pour plus de déterminisme aux tentatives suivantes
                    adjusted_temp = max(0.0, self.temperature - 0.1 * retries)
                    payload["temperature"] = adjusted_temp
                    logger.info(f"Température ajustée à {adjusted_temp} pour la tentative {retries}")
                
                # Effectuer la requête avec timeout
                response = requests.post(api_url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                
                # Analyser la réponse
                response_data = response.json()
                logger.debug(f"Réponse reçue de LM Studio")
                
                if "choices" not in response_data or not response_data["choices"]:
                    logger.error("La réponse de LM Studio ne contient pas de choix valides")
                    if retries < self.max_retries:
                        retries += 1
                        time.sleep(self.retry_delay)
                        continue
                    return {"error": "invalid_response", "raw_response": str(response_data)}
                    
                result = response_data["choices"][0].get("message", {}).get("content", "").strip()
                
                # Si le résultat est vide, réessayer
                if not result:
                    logger.error("LM Studio a retourné une réponse vide")
                    if retries < self.max_retries:
                        retries += 1
                        time.sleep(self.retry_delay)
                        continue
                    return {"error": "empty_response"}
                
                # Si format JSON demandé, parser et valider la réponse
                if output_format.lower() == "json":
                    return self._process_json_response(result)
                
                # Pour les autres formats, retourner le résultat brut
                return {"raw_response": result}
                
            except requests.exceptions.Timeout:
                logger.error(f"Timeout lors de la connexion à LM Studio (après {self.timeout}s)")
                if retries < self.max_retries:
                    # Augmenter le timeout pour la prochaine tentative
                    self.timeout = int(self.timeout * 1.5)
                    logger.info(f"Augmentation du timeout à {self.timeout}s pour la prochaine tentative")
                    retries += 1
                    time.sleep(self.retry_delay)
                    continue
                return {"error": "timeout", "message": f"Délai d'attente dépassé ({self.timeout}s)"}
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Erreur de connexion à LM Studio: {str(e)}")
                if retries < self.max_retries:
                    retries += 1
                    time.sleep(self.retry_delay * 2)  # Attente plus longue pour les erreurs de connexion
                    continue
                return {"error": "connection_error", "message": str(e)}
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur de requête à l'API LM Studio: {str(e)}")
                if retries < self.max_retries:
                    retries += 1
                    time.sleep(self.retry_delay)
                    continue
                return {"error": "request_error", "message": str(e)}
                
            except Exception as e:
                logger.error(f"Erreur générale lors de l'appel à LM Studio: {str(e)}")
                if retries < self.max_retries:
                    retries += 1
                    time.sleep(self.retry_delay)
                    continue
                return {"error": "general_error", "message": str(e)}
    
    def _process_json_response(self, result: str) -> Dict[str, Any]:
        """
        Traite et valide une réponse JSON.
        
        Args:
            result (str): Résultat brut de l'API
            
        Returns:
            Dict[str, Any]: JSON parsé ou dictionnaire d'erreur
        """
        # Nettoyage préliminaire de la réponse pour extraction de JSON
        json_str = result
        
        # Si le résultat contient un bloc code markdown, l'extraire
        if "```json" in json_str:
            parts = json_str.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0]
                json_str = json_part.strip()
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) > 1:
                json_str = parts[1].strip()
        
        # Extraction plus agressive: chercher à partir du premier {
        first_brace = json_str.find("{")
        if first_brace >= 0:
            last_brace = json_str.rfind("}")
            if last_brace > first_brace:
                json_str = json_str[first_brace:last_brace+1]
        
        # Nettoyage final pour éliminer caractères parasites
        json_str = json_str.strip()
        if json_str and not json_str.startswith("{"):
            logger.warning(f"La réponse ne commence pas par '{{': {json_str[:50]}...")
        
        try:
            # Tenter de parser le JSON
            parsed_json = json.loads(json_str)
            logger.info("JSON parsé avec succès")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Erreur de décodage JSON: {str(e)}. Tentative de réparation...")
            
            # Tentative de réparation basique du JSON
            try:
                from json.decoder import JSONDecodeError
                import re
                
                # Remplacer les simples quotes par des doubles quotes pour les clés
                fixed_json = re.sub(r"'([^']+)':", r'"\1":', json_str)
                # Ajouter des doubles quotes pour les valeurs simples quotes
                fixed_json = re.sub(r': \'([^\']+)\'', r': "\1"', fixed_json)
                # Réparer les virgules manquantes entre les objets d'un tableau
                fixed_json = re.sub(r'}\s*{', '},{', fixed_json)
                # Réparer les guillemets non fermés
                fixed_json = re.sub(r'": "([^"]*?)(\s*[,}])', r'": "\1"\2', fixed_json)
                
                parsed_json = json.loads(fixed_json)
                logger.info("JSON réparé et parsé avec succès")
                return parsed_json
            except Exception:
                pass
            
            # En dernier recours, essayer de construire un JSON minimal
            try:
                # Extraire ce qui semble être des titres, dates, etc.
                import re
                titres = re.findall(r'"titre"?\s*:?\s*"([^"]+)"', json_str)
                dates = re.findall(r'"date"?\s*:?\s*"([^"]+)"', json_str)
                auteurs = re.findall(r'"auteur"?\s*:?\s*"([^"]+)"', json_str)
                
                minimal_json = {}
                if titres:
                    minimal_json["titres"] = titres
                if dates:
                    minimal_json["dates"] = dates
                if auteurs:
                    minimal_json["auteurs"] = auteurs
                    
                if minimal_json:
                    logger.info("Construit un JSON minimal à partir des données extraites")
                    return minimal_json
            except:
                pass
                
            # En cas d'échec, retourner le résultat brut
            return {"error": "json_parse_error", "raw_response": result}
