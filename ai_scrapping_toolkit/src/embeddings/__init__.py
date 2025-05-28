"""
Module pour la vectorisation et la recherche sémantique.
"""

from .vector_db import (
    chunks_to_embeddings, 
    create_faiss_index, 
    save_faiss_index, 
    load_faiss_index,
    search_similar
)

__all__ = [
    'chunks_to_embeddings', 
    'create_faiss_index', 
    'save_faiss_index', 
    'load_faiss_index',
    'search_similar'
]
