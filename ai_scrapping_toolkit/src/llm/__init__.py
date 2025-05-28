"""
Module pour l'intégration avec les modèles de langage.
"""

from .providers import get_llm_provider
from .data_extractor import extract_data_from_chunks, aggregate_extraction_results

# Importer le module enhanced seulement s'il existe
try:
    from .enhanced_data_extractor import enhanced_extract_data_from_chunks, EnhancedDataExtractor
    __all__ = [
        'get_llm_provider',
        'extract_data_from_chunks',
        'aggregate_extraction_results',
        'enhanced_extract_data_from_chunks',
        'EnhancedDataExtractor'
    ]
except ImportError:
    __all__ = [
        'get_llm_provider',
        'extract_data_from_chunks',
        'aggregate_extraction_results'
    ]
