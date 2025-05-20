"""
Module pour l'interaction avec des modèles de langage (LLM).
"""

from .extraction import extract_data_from_chunks, aggregate_extraction_results
from .providers import get_llm_provider

__all__ = [
    'extract_data_from_chunks', 
    'aggregate_extraction_results',
    'get_llm_provider'
]
