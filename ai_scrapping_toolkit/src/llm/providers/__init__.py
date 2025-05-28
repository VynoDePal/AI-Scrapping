"""
Providers pour différents modèles de langage.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

def get_llm_provider(provider_name="openai", **config):
    """
    Renvoie un provider LLM selon le nom spécifié.
    
    Args:
        provider_name (str): Nom du provider ("openai", "ollama", "huggingface", "lmstudio", "openrouter")
        config: Configuration spécifique au provider
        
    Returns:
        object: Instance du provider LLM
    """
    provider_mapping = {
        "openai": "OpenAIProvider",
        "ollama": "OllamaProvider",
        "huggingface": "HuggingFaceProvider",
        "lmstudio": "LMStudioProvider",
        "openrouter": "OpenRouterProvider"  # Ajout du nouveau provider
    }
    
    if provider_name not in provider_mapping:
        logger.error(f"Provider '{provider_name}' non supporté. Utilisation d'OpenAI par défaut.")
        provider_name = "openai"
    
    provider_class_name = provider_mapping[provider_name]
    
    try:
        # Importer dynamiquement le module du provider
        module = importlib.import_module(f"src.llm.providers.{provider_name}_provider")
        provider_class = getattr(module, provider_class_name)
        return provider_class(**config)
    except (ImportError, AttributeError) as e:
        logger.error(f"Erreur lors du chargement du provider '{provider_name}': {str(e)}")
        
        # Essayer de charger OpenAI comme fallback
        if provider_name != "openai":
            logger.warning("Tentative d'utiliser OpenAI comme fallback...")
            try:
                module = importlib.import_module("src.llm.providers.openai_provider")
                provider_class = getattr(module, "OpenAIProvider")
                return provider_class(**config)
            except (ImportError, AttributeError):
                logger.error("Impossible de charger le provider OpenAI de secours.")
        
        raise ImportError(f"Impossible de charger le provider LLM '{provider_name}'")
