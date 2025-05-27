"""
Module source principal du AI Scrapping Toolkit.
"""

# Import des modules principaux
from .scrapers import fetch_content
from .processors import (
    preprocess_html, 
    extract_main_content, 
    get_page_title, 
    html_to_chunks, 
    pdf_to_chunks,
    semantic_html_to_chunks
)
from .embeddings import chunks_to_embeddings, create_faiss_index, save_faiss_index
from .llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results

__all__ = [
    'fetch_content',
    'preprocess_html',
    'extract_main_content', 
    'get_page_title',
    'html_to_chunks',
    'pdf_to_chunks',
    'semantic_html_to_chunks',
    'chunks_to_embeddings',
    'create_faiss_index',
    'save_faiss_index',
    'get_llm_provider',
    'extract_data_from_chunks',
    'aggregate_extraction_results'
]
