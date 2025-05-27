"""
Module pour la vectorisation et la recherche sémantique.
"""

from .embedding_generator import chunks_to_embeddings
from .faiss_manager import create_faiss_index, save_faiss_index, load_faiss_index, search_similar

__all__ = [
    'chunks_to_embeddings',
    'create_faiss_index',
    'save_faiss_index',
    'load_faiss_index',
    'search_similar'
]
