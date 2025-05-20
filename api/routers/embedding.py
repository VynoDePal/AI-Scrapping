"""
Routeur pour les fonctionnalités de vectorisation et recherche.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form, Path
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import tempfile
import os
import json
import uuid
import datetime
import base64

from ..models.requests import VectorizationRequest, SearchRequest
from ..models.responses import VectorizationResponse, SearchResponse, SearchResultItem, ErrorResponse
from src.embeddings import chunks_to_embeddings, create_faiss_index, save_faiss_index, load_faiss_index, search_similar

router = APIRouter()
logger = logging.getLogger(__name__)

# Répertoire pour stocker les index temporaires
TEMP_INDEX_DIR = "/tmp/aiscrapping_indexes"
os.makedirs(TEMP_INDEX_DIR, exist_ok=True)


@router.post(
    "/vectorize",
    response_model=VectorizationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def vectorize_chunks(request: VectorizationRequest):
    """
    Vectorise une liste de chunks de texte et crée un index FAISS.
    
    - **chunks**: Liste de chunks à vectoriser
    - **model_name**: Modèle sentence-transformers à utiliser
    - **index_type**: Type d'index FAISS
    """
    try:
        logger.info(f"Démarrage de la vectorisation avec le modèle {request.model_name}")
        
        if not request.chunks:
            raise HTTPException(status_code=400, detail="Aucun chunk fourni pour la vectorisation")
        
        # Générer les embeddings
        embeddings = chunks_to_embeddings(request.chunks, model_name=request.model_name)
        
        if len(embeddings) == 0:
            raise HTTPException(status_code=400, detail="Échec de la génération des embeddings")
        
        # Créer l'index FAISS
        index, index_metadata = create_faiss_index(embeddings, request.chunks, request.index_type)
        
        # Sauvegarder l'index temporairement
        index_id = str(uuid.uuid4())
        index_path = os.path.join(TEMP_INDEX_DIR, index_id)
        save_faiss_index(index, index_metadata, index_path)
        
        return VectorizationResponse(
            index_id=index_id,
            vector_count=len(embeddings),
            vector_dimension=len(embeddings[0]),
            model_name=request.model_name,
            index_type=request.index_type,
            chunks=request.chunks
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la vectorisation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de vectorisation: {str(e)}")


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def search_in_index(request: SearchRequest):
    """
    Recherche dans un index vectoriel avec une requête en langage naturel.
    
    - **query**: Requête de recherche
    - **index_path**: Chemin ou ID de l'index
    - **top_k**: Nombre de résultats à retourner
    - **model_name**: Modèle sentence-transformers à utiliser
    """
    try:
        logger.info(f"Recherche: '{request.query}' dans l'index {request.index_path}")
        
        # Déterminer si l'index_path est un ID temporaire ou un chemin
        if os.path.exists(os.path.join(TEMP_INDEX_DIR, request.index_path + ".index")):
            index_path = os.path.join(TEMP_INDEX_DIR, request.index_path)
        elif os.path.exists(request.index_path + ".index"):
            index_path = request.index_path
        else:
            raise HTTPException(status_code=400, detail=f"Index {request.index_path} introuvable")
        
        # Charger l'index
        index, index_metadata = load_faiss_index(index_path)
        
        # Effectuer la recherche
        search_results = search_similar(
            query=request.query,
            index=index,
            index_metadata=index_metadata,
            model_name=request.model_name,
            top_k=request.top_k
        )
        
        # Convertir les résultats au format requis
        result_items = [
            SearchResultItem(
                score=result['score'],
                index=result['index'],
                chunk=result['chunk'],
                metadata=result.get('metadata')
            )
            for result in search_results
        ]
        
        return SearchResponse(
            query=request.query,
            results=result_items,
            result_count=len(result_items),
            model_name=request.model_name
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de recherche: {str(e)}")
