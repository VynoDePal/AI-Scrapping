"""
Routeur pour les fonctionnalités d'export de données.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import time
import datetime
import tempfile
import os
import base64
import pandas as pd
import json

from ..models.requests import ExportRequest
from ..models.responses import ExportResponse, ErrorResponse
from src.utils.file_handler import load_file

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=ExportResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def export_data(request: ExportRequest):
    """
    Exporte des données au format spécifié (CSV, JSON, Excel).
    
    - **data**: Données à exporter
    - **output_format**: Format d'export (csv, json, excel)
    - **remove_duplicates**: Supprimer les doublons
    - **date_columns**: Liste des colonnes de date
    - **date_format**: Format des dates
    - **sort_by**: Colonne pour le tri
    - **sort_ascending**: Tri ascendant
    - **columns**: Liste des colonnes à inclure
    - **flatten_complex**: Aplatir les structures complexes
    """
    try:
        logger.info(f"Export au format {request.output_format}")
        
        # Créer un DataFrame pandas
        df = pd.DataFrame(request.data)
        
        # Nettoyer les données
        if request.remove_duplicates:
            df = df.drop_duplicates()
        
        # Filtrer les colonnes si spécifié
        if request.columns:
            valid_columns = [col for col in request.columns if col in df.columns]
            if valid_columns:
                df = df[valid_columns]
        
        # Convertir les dates si spécifié
        if request.date_columns:
            for col in request.date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if request.date_format:
                        df[col] = df[col].dt.strftime(request.date_format)
        
        # Trier si spécifié
        if request.sort_by and request.sort_by in df.columns:
            df = df.sort_values(by=request.sort_by, ascending=request.sort_ascending)
        
        # Exporter selon le format demandé
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            if request.output_format == "csv":
                df.to_csv(tmp.name, index=request.include_index)
                content_type = "text/csv"
            elif request.output_format == "excel":
                df.to_excel(tmp.name, index=request.include_index)
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:  # json
                df.to_json(tmp.name, orient='records', date_format='iso')
                content_type = "application/json"
        
            # Lire le fichier et l'encoder en base64
            tmp.seek(0)
            with open(tmp.name, 'rb') as f:
                file_content = base64.b64encode(f.read()).decode('utf-8')
        
        # Supprimer le fichier temporaire
        os.unlink(tmp.name)
        
        return ExportResponse(
            file_format=request.output_format,
            row_count=len(df),
            column_count=len(df.columns),
            columns=df.columns.tolist(),
            file_content_base64=file_content,
            content_type=content_type
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'export: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur d'export: {str(e)}")


@router.post(
    "/file",
    response_model=ExportResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def export_from_json_file(
    file: UploadFile = File(...),
    output_format: str = Form("csv"),
    remove_duplicates: bool = Form(True),
    include_index: bool = Form(False),
    date_format: Optional[str] = Form(None),
    sort_by: Optional[str] = Form(None),
    sort_ascending: bool = Form(True)
):
    """
    Exporte un fichier JSON en CSV, Excel ou JSON restructuré.
    """
    try:
        # Lire le contenu du fichier
        content = await file.read()
        data = json.loads(content)
        
        # Créer la requête d'export
        request = ExportRequest(
            data=data,
            output_format=output_format,
            remove_duplicates=remove_duplicates,
            date_format=date_format or "%Y-%m-%d",
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            include_index=include_index,
            flatten_complex=True
        )
        
        # Utiliser la fonction existante
        return await export_data(request)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Fichier JSON invalide")
    except Exception as e:
        logger.error(f"Erreur lors du traitement du fichier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement du fichier: {str(e)}")
