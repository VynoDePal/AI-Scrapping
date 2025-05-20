"""
Routeur pour les fonctionnalités de scraping web.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Optional
import logging
import time
import datetime

from ..models.requests import ScrapingRequest
from ..models.responses import ScrapingResponse, ErrorResponse
from src.scrapers import fetch_content
from src.processors import preprocess_html, extract_main_content, get_page_title

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=ScrapingResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def scrape_url(request: ScrapingRequest):
    """
    Récupère le contenu d'une URL avec différentes options de traitement.
    
    - **url**: URL du site à scraper
    - **method**: Méthode de récupération (requests, selenium, auto)
    - **wait_time**: Temps d'attente pour Selenium (secondes)
    - **preprocess**: Prétraiter le HTML pour extraire le texte pertinent
    - **extract_main_content**: Extraire uniquement le contenu principal
    - **respect_robots**: Respecter les règles robots.txt
    - **user_agent**: User-Agent à utiliser pour les requêtes
    - **rate_limit**: Délai minimum entre les requêtes (secondes)
    """
    try:
        logger.info(f"Démarrage du scraping pour {request.url}")
        
        # Récupérer le contenu HTML
        content = fetch_content(
            str(request.url),
            method=request.method,
            wait_time=request.wait_time,
            respect_robots=request.respect_robots,
            user_agent=request.user_agent,
            rate_limit=request.rate_limit
        )
        
        if not content:
            raise HTTPException(status_code=400, detail="Impossible de récupérer le contenu")
        
        # Prétraiter ou extraire le contenu principal si demandé
        content_type = "HTML"
        if request.extract_main_content:
            content = extract_main_content(content) or content
            content_type = "main_content"
        elif request.preprocess:
            content = preprocess_html(content)
            content_type = "preprocessed"
        
        # Obtenir le titre de la page
        title = get_page_title(content)
        
        # Préparer la réponse
        return ScrapingResponse(
            content=content,
            url=str(request.url),
            title=title,
            content_length=len(content),
            content_type=content_type,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du scraping de {request.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de scraping: {str(e)}")


@router.post(
    "/check",
    response_model=dict,
    responses={400: {"model": ErrorResponse}},
)
async def check_url_access(url: str, check_robots: bool = True):
    """
    Vérifie si une URL est accessible et si elle peut être scrapée selon robots.txt.
    """
    from src.scrapers.robots_checker import RobotsChecker
    
    try:
        checker = RobotsChecker(respect_robots=check_robots)
        can_fetch, reason = checker.can_fetch(url)
        
        return {
            "url": url,
            "accessible": can_fetch,
            "reason": reason,
            "checked_robots": check_robots,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la vérification: {str(e)}")
