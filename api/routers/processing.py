"""
Routeur pour les fonctionnalités de traitement de contenu.
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import time
import datetime
import statistics

from ..models.requests import ChunkingRequest, ProcessingRequest
from ..models.responses import ChunkingResponse, ProcessingResponse, ErrorResponse
from src.processors import html_to_chunks
from src.processors.data_processor import filter_by_date, analyze_sentiment, categorize_text, sort_and_filter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/chunk",
    response_model=ChunkingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def create_chunks(request: ChunkingRequest):
    """
    Découpe un contenu HTML en chunks selon les paramètres spécifiés.
    
    - **content**: Contenu HTML à découper
    - **chunk_method**: Méthode de découpage (tags, length, hybrid)
    - **max_length**: Taille maximale d'un chunk
    - **overlap**: Chevauchement entre chunks
    """
    try:
        logger.info(f"Démarrage du chunking (méthode: {request.chunk_method})")
        
        # Générer les chunks
        chunks = html_to_chunks(
            request.content,
            method=request.chunk_method,
            max_length=request.max_length,
            overlap=request.overlap
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Échec du chunking: aucun chunk généré")
        
        # Calculer la taille moyenne des chunks
        chunk_sizes = [len(chunk) for chunk in chunks]
        avg_size = int(statistics.mean(chunk_sizes))
        
        return ChunkingResponse(
            chunks=chunks,
            chunk_count=len(chunks),
            average_chunk_size=avg_size,
            method=request.chunk_method
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du chunking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de chunking: {str(e)}")


@router.post(
    "/file/chunk",
    response_model=ChunkingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chunk_upload_file(
    file: UploadFile = File(...),
    chunk_method: str = Form("hybrid"),
    max_length: int = Form(1000),
    overlap: int = Form(100)
):
    """
    Découpe le contenu d'un fichier HTML en chunks.
    """
    try:
        # Lire le contenu du fichier
        content = await file.read()
        content_str = content.decode("utf-8")
        
        # Créer la requête de chunking
        request = ChunkingRequest(
            content=content_str,
            chunk_method=chunk_method,
            max_length=max_length,
            overlap=overlap
        )
        
        # Utiliser la fonction existante
        return await create_chunks(request)
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du fichier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement du fichier: {str(e)}")


@router.post(
    "/data",
    response_model=ProcessingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def process_data(request: ProcessingRequest):
    """
    Traite des données extraites selon différents critères.
    
    Permet de:
    - Filtrer les données par date
    - Analyser le sentiment des textes
    - Catégoriser les textes
    - Trier et filtrer les résultats
    """
    try:
        data = request.data
        operations = []
        original_count = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
        
        # Filtrer par date si demandé
        if request.filter_date and request.date_field:
            data = filter_by_date(
                data,
                date_field=request.date_field,
                days=request.days,
                start_date=request.start_date,
                end_date=request.end_date
            )
            operations.append(f"filter_date({request.date_field}, days={request.days})")
        
        # Analyser le sentiment si demandé
        if request.analyze_sentiment and request.sentiment_field:
            data = analyze_sentiment(
                data,
                text_field=request.sentiment_field,
                provider=request.sentiment_provider or "huggingface"
            )
            operations.append(f"analyze_sentiment({request.sentiment_field})")
        
        # Catégoriser les textes si demandé
        if request.categorize and request.category_field:
            data = categorize_text(
                data,
                text_field=request.category_field,
                categories=request.categories
            )
            operations.append(f"categorize({request.category_field})")
        
        # Trier et filtrer si demandé
        if request.sort_by or request.filter_expr:
            data = sort_and_filter(
                data,
                sort_by=request.sort_by,
                ascending=not request.sort_desc,
                filter_expr=request.filter_expr
            )
            operations.append(f"sort_and_filter(sort_by={request.sort_by}, filter={request.filter_expr})")
        
        # Calculer le nombre d'éléments filtrés
        current_count = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
        filtered_count = original_count - current_count if original_count > current_count else 0
        
        return ProcessingResponse(
            data=data,
            operations=operations,
            filtered_count=filtered_count if filtered_count > 0 else None,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement des données: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")
