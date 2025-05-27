"""
Module pour l'intégration avec les modèles de langage.
"""

from .providers import get_llm_provider
from .data_extractor import extract_data_from_chunks, aggregate_extraction_results

__all__ = [
    'get_llm_provider',
    'extract_data_from_chunks',
    'aggregate_extraction_results'
]
