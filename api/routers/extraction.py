"""
Routeur pour les fonctionnalités d'extraction de données avec les LLMs.
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import time
import datetime
import json
import os  # Ajout de l'import manquant
import tempfile  # Utile pour la gestion des fichiers temporaires

from ..models.requests import ExtractionRequest
from ..models.responses import ExtractionResponse, ErrorResponse
from src.llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results
from src.processors import html_to_chunks, pdf_to_chunks

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=ExtractionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def extract_data(request: ExtractionRequest):
    """
    Extrait des données structurées à partir d'un contenu HTML en utilisant des modèles de langage.
    """
    try:
        logger.info(f"Extraction avec {request.provider}/{request.model}: '{request.query}'")
        
        # Découper le contenu en chunks (utiliser semantic par défaut pour de meilleurs résultats)
        chunk_method = request.chunk_method if request.chunk_method != "hybrid" else "semantic"
        chunks = html_to_chunks(
            request.content, 
            method=chunk_method,
            max_length=request.chunk_size
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Échec du chunking: aucun chunk généré")
        
        # Limiter le nombre de chunks si spécifié
        if request.max_chunks and request.max_chunks > 0 and len(chunks) > request.max_chunks:
            chunks = chunks[:request.max_chunks]
        
        # Initialiser le provider LLM
        llm_config = {
            "api_key": request.api_key,
            "model": request.model,
            "temperature": request.temperature
        }
        
        if request.host:
            llm_config["host"] = request.host
        
        llm_provider = get_llm_provider(request.provider, **llm_config)
        
        # Extraire les données avec le mode amélioré par défaut
        extraction_results = extract_data_from_chunks(
            chunks=chunks,
            query=request.query,
            llm_provider=llm_provider,
            max_workers=min(4, len(chunks)),
            enhanced_mode=True,  # Activer le mode amélioré
            url=getattr(request, 'source_url', '')  # URL source si disponible
        )
        
        # Le mode amélioré retourne déjà un résultat agrégé
        if extraction_results and isinstance(extraction_results[0], dict):
            aggregated_data = extraction_results[0]
        else:
            # Fallback vers l'agrégation classique
            aggregated_data = aggregate_extraction_results(extraction_results)
        
        return ExtractionResponse(
            query=request.query,
            data=aggregated_data,
            provider=request.provider,
            model=request.model,
            chunk_count=len(chunks),
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur d'extraction: {str(e)}")


@router.post(
    "/file",
    response_model=ExtractionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def extract_from_file(
    file: UploadFile = File(...),
    query: str = Form(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo"),
    chunk_size: int = Form(4000),
    chunk_method: str = Form("hybrid"),
    max_chunks: Optional[int] = Form(None),
    api_key: Optional[str] = Form(None),
    temperature: float = Form(0.0)
):
    """
    Extrait des données structurées à partir d'un fichier HTML.
    """
    try:
        # Lire le contenu du fichier
        content = await file.read()
        content_str = content.decode("utf-8")
        
        # Créer la requête d'extraction
        request = ExtractionRequest(
            content=content_str,
            query=query,
            provider=provider,
            model=model,
            chunk_size=chunk_size,
            chunk_method=chunk_method,
            max_chunks=max_chunks,
            api_key=api_key,
            temperature=temperature
        )
        
        # Utiliser la fonction existante
        return await extract_data(request)
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du fichier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement du fichier: {str(e)}")


@router.post(
    "/pdf",
    response_model=ExtractionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def extract_from_pdf(
    file: UploadFile = File(...),
    query: str = Form(...),
    provider: str = Form("openai"),
    model: str = Form("gpt-3.5-turbo"),
    chunk_method: str = Form("pages"),
    chunk_size: int = Form(4000),
    max_chunks: Optional[int] = Form(None),
    api_key: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    auto_enhance: bool = Form(True),  # Nouvelle option pour améliorer automatiquement les requêtes
):
    """
    Extrait des données structurées à partir d'un fichier PDF en utilisant des modèles de langage.
    
    - **file**: Fichier PDF à analyser
    - **query**: Requête d'extraction en langage naturel
    - **provider**: Fournisseur du modèle LLM
    - **model**: Modèle LLM à utiliser
    - **chunk_method**: Méthode de découpage du PDF (pages, paragraphs, length)
    - **chunk_size**: Taille maximale des chunks
    - **max_chunks**: Nombre maximal de chunks à traiter
    - **auto_enhance**: Améliorer automatiquement les requêtes génériques
    """
    temp_file_path = ""
    try:
        # Sauvegarder temporairement le fichier PDF
        temp_file_path = f"/tmp/temp_pdf_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Extraire les métadonnées du PDF pour détection du type de document
        from src.processors.pdf_processor import extract_pdf_metadata
        metadata = extract_pdf_metadata(temp_file_path)
        logger.info(f"Métadonnées du PDF: {metadata}")
        
        # Améliorer la requête si elle est générique et que l'auto-amélioration est activée
        original_query = query
        if auto_enhance and (query.lower() in ["string", "extract", "extraire", "information", ""]):
            # Détecter le type de document en fonction du contenu et du nom de fichier
            is_resume = any(term in file.filename.lower() for term in ["resume", "cv", "curriculum"])
            is_invoice = any(term in file.filename.lower() for term in ["invoice", "facture", "receipt"])
            is_article = any(term in file.filename.lower() for term in ["article", "paper", "publication"])
            
            # Extraire un échantillon du contenu pour détection
            from src.processors.pdf_processor import extract_text_from_pdf
            sample_text = extract_text_from_pdf(temp_file_path)[:1000].lower()
            
            # Affiner la détection basée sur le contenu
            if not is_resume and any(term in sample_text for term in ["resume", "cv", "skills", "experience", "education", "compétences", "expérience"]):
                is_resume = True
            if not is_invoice and any(term in sample_text for term in ["invoice", "facture", "total", "payment", "tax", "tva", "montant"]):
                is_invoice = True
            if not is_article and any(term in sample_text for term in ["abstract", "introduction", "conclusion", "references", "keywords"]):
                is_article = True
            
            # Construire une requête améliorée en fonction du type détecté
            if is_resume:
                query = """Extraire et structurer les informations suivantes du CV:
                - informations_personnelles: nom complet, email, téléphone, adresse, site web, profil LinkedIn
                - résumé_professionnel: texte du résumé ou introduction
                - compétences: liste des compétences techniques et personnelles
                - expérience_professionnelle: liste des postes avec entreprise, période, titre, description
                - formation: liste des formations avec établissement, diplôme, période, description
                - langues: langues parlées et niveau
                - certifications: liste des certifications pertinentes
                - projets: projets significatifs mentionnés
                
                Regroupe ces informations dans une structure JSON cohérente."""
                
            elif is_invoice:
                query = """Extraire et structurer les informations suivantes de la facture:
                - informations_émetteur: nom, adresse, numéro de téléphone, email
                - informations_client: nom, adresse, identifiant client
                - détails_facture: numéro de facture, date d'émission, date d'échéance
                - articles: liste des articles/services avec description, quantité, prix unitaire et total
                - montants: sous-total, taxes (TVA ou autres), frais supplémentaires, total
                - modalités_paiement: méthode de paiement, coordonnées bancaires
                
                Regroupe ces informations dans une structure JSON cohérente."""
                
            elif is_article:
                query = """Extraire et structurer les informations suivantes de l'article:
                - méta_informations: titre, auteurs, date de publication, journal/conférence
                - résumé: résumé ou abstract complet
                - structure_principale: introduction, méthodologie, résultats, discussion, conclusion
                - mots_clés: liste des mots-clés
                - références: liste des références bibliographiques principales
                
                Regroupe ces informations dans une structure JSON cohérente."""
                
            else:
                # Requête générale pour tout type de document
                query = """Analyse ce document et extrait les informations clés suivantes:
                - titre: titre principal du document
                - auteur: auteur ou créateur du document
                - date: date de création ou de publication
                - type_document: type de document détecté
                - sections_principales: liste des principales sections avec leur contenu résumé
                - points_clés: liste des informations importantes extraites
                - entités: personnes, organisations, lieux et dates mentionnés
                
                Regroupe ces informations dans une structure JSON cohérente et détaillée."""
                
            logger.info(f"Requête améliorée automatiquement: '{query[:50]}...' (basée sur la détection de type de document)")
        
        # Extraire les chunks du PDF
        chunks = pdf_to_chunks(
            temp_file_path,
            method=chunk_method,
            max_length=chunk_size,
            overlap=100  # valeur par défaut
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Échec de l'extraction du PDF: aucun contenu récupéré")
        
        logger.info(f"PDF traité avec succès: {len(chunks)} chunks extraits")
        
        # Limiter le nombre de chunks si spécifié
        if max_chunks and max_chunks > 0 and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
        
        # Initialiser le provider LLM
        llm_config = {
            "api_key": api_key,
            "model": model,
            "temperature": temperature
        }
        
        llm_provider = get_llm_provider(provider, **llm_config)
        
        # Extraire les données des chunks
        extraction_results = extract_data_from_chunks(
            chunks=chunks,
            query=query,
            llm_provider=llm_provider,
            max_workers=min(4, len(chunks))
        )
        
        # Agréger les résultats
        aggregated_data = aggregate_extraction_results(extraction_results)
        
        # Supprimer le fichier temporaire
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return ExtractionResponse(
            query=original_query + (" (requête améliorée automatiquement)" if original_query != query else ""),
            data=aggregated_data,
            provider=provider,
            model=model,
            chunk_count=len(chunks),
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du PDF: {str(e)}")
        # Assurer que le fichier temporaire est supprimé en cas d'erreur
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Erreur d'extraction PDF: {str(e)}")
