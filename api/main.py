"""
Point d'entrée principal de l'API FastAPI pour AI Scrapping.
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import logging
import time
import os

# Importer les routeurs
from .routers import scraping, processing, embedding, extraction, export

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Créer l'application FastAPI
app = FastAPI(
    title="AI Scrapping API",
    description="""
    API pour extraire, prétraiter et traiter des contenus web avec des modèles d'IA.
    
    Fonctionnalités:
    - Scraping de sites web (avec respect de robots.txt)
    - Prétraitement et extraction de contenu principal
    - Segmentation en chunks pour les modèles d'IA
    - Vectorisation et recherche sémantique
    - Extraction de données structurées via des LLMs
    - Traitement avancé des données extraites
    - Export de données en CSV
    """,
    version="1.0.0",
    docs_url=None,  # Désactiver la documentation par défaut
    redoc_url=None,  # Désactiver ReDoc
)

# Configuration CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines (à restreindre en production)
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes
    allow_headers=["*"],  # Autoriser tous les headers
)

# Middleware pour la journalisation et la gestion du rate limiting
@app.middleware("http")
async def log_and_rate_limit(request: Request, call_next):
    """Middleware pour la journalisation et limitation du débit."""
    client_host = request.client.host
    logger.info(f"Requête de {client_host}: {request.method} {request.url}")
    
    # Rate limiting très simple (à améliorer avec Redis en production)
    time.sleep(0.1)  
    
    # Traiter la requête
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Ajouter le temps de traitement aux headers
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Requête traitée en {process_time:.4f}s")
    
    return response

# Handler global pour les erreurs
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'exceptions global."""
    logger.error(f"Erreur lors du traitement de {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Erreur serveur: {str(exc)}"}
    )

# Route pour la documentation Swagger personnalisée
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Page de documentation Swagger personnalisée."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AI Scrapping API - Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

# Vérification de l'initialisation de l'environnement
@app.on_event("startup")
async def startup_event():
    """Vérification de l'environnement au démarrage."""
    logger.info("Démarrage de l'API AI Scrapping...")
    
    # Vérifier si les variables d'environnement clés sont définies
    env_vars = ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]
    for var in env_vars:
        if var in os.environ:
            logger.info(f"Variable d'environnement {var} trouvée.")
        else:
            logger.warning(f"Variable d'environnement {var} non définie.")

# Page d'accueil API
@app.get("/", tags=["Général"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "name": "AI Scrapping API",
        "version": "1.0.0",
        "description": "API pour l'extraction et le traitement de contenu web avec l'IA",
        "documentation": "/docs"
    }

# Vérification de la santé de l'API
@app.get("/health", tags=["Général"])
async def health_check():
    """Vérification de santé de l'API."""
    return {"status": "healthy"}

# Inclure les routeurs
app.include_router(scraping.router, prefix="/scraping", tags=["Scraping"])
app.include_router(processing.router, prefix="/processing", tags=["Traitement de contenu"])
app.include_router(embedding.router, prefix="/embedding", tags=["Vectorisation et recherche"])
app.include_router(extraction.router, prefix="/extraction", tags=["Extraction de données"])
app.include_router(export.router, prefix="/export", tags=["Export de données"])

# Avertir si l'API est exécutée directement
if __name__ == "__main__":
    logger.warning(
        "Exécution directe du fichier main.py. "
        "Utilisez plutôt uvicorn: 'uvicorn api.main:app --reload'"
    )
