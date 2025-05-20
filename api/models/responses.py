"""
Modèles Pydantic pour les réponses API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union


class ErrorResponse(BaseModel):
    """Modèle pour les réponses d'erreur."""
    detail: str = Field(..., description="Message d'erreur")


class ScrapingResponse(BaseModel):
    """Modèle pour les réponses de scraping."""
    content: str = Field(..., description="Contenu HTML récupéré")
    url: str = Field(..., description="URL du site scrapé")
    title: Optional[str] = Field(None, description="Titre de la page")
    content_length: int = Field(..., description="Longueur du contenu")
    content_type: str = Field(..., description="Type de contenu (HTML, texte, etc.)")
    timestamp: str = Field(..., description="Horodatage de la requête")


class ChunkingResponse(BaseModel):
    """Modèle pour les réponses de chunking."""
    chunks: List[str] = Field(..., description="Liste des chunks générés")
    chunk_count: int = Field(..., description="Nombre de chunks")
    average_chunk_size: int = Field(..., description="Taille moyenne des chunks")
    method: str = Field(..., description="Méthode de chunking utilisée")


class VectorizationResponse(BaseModel):
    """Modèle pour les réponses de vectorisation."""
    index_id: str = Field(..., description="Identifiant de l'index vectoriel")
    vector_count: int = Field(..., description="Nombre de vecteurs dans l'index")
    vector_dimension: int = Field(..., description="Dimension des vecteurs")
    model_name: str = Field(..., description="Modèle utilisé pour la vectorisation")
    index_type: str = Field(..., description="Type d'index FAISS")
    chunks: List[str] = Field(..., description="Liste des chunks vectorisés")


class SearchResultItem(BaseModel):
    """Élément de résultat de recherche."""
    score: float = Field(..., description="Score de similarité")
    index: int = Field(..., description="Index dans la base de données")
    chunk: str = Field(..., description="Contenu du chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Métadonnées associées")


class SearchResponse(BaseModel):
    """Modèle pour les réponses de recherche."""
    query: str = Field(..., description="Requête de recherche")
    results: List[SearchResultItem] = Field(..., description="Résultats de recherche")
    result_count: int = Field(..., description="Nombre de résultats")
    model_name: str = Field(..., description="Modèle utilisé pour la recherche")


class ExtractionResponse(BaseModel):
    """Modèle pour les réponses d'extraction de données."""
    query: str = Field(..., description="Requête d'extraction")
    data: Dict[str, Any] = Field(..., description="Données extraites")
    provider: str = Field(..., description="Provider utilisé")
    model: str = Field(..., description="Modèle utilisé")
    chunk_count: int = Field(..., description="Nombre de chunks traités")
    timestamp: str = Field(..., description="Horodatage de la requête")


class ProcessingResponse(BaseModel):
    """Modèle pour les réponses de traitement de données."""
    data: Dict[str, Any] = Field(..., description="Données traitées")
    operations: List[str] = Field(..., description="Opérations effectuées")
    filtered_count: Optional[int] = Field(None, description="Nombre d'éléments filtrés")
    timestamp: str = Field(..., description="Horodatage de la requête")


class ExportResponse(BaseModel):
    """Modèle pour les réponses d'export de données."""
    file_format: str = Field(..., description="Format du fichier")
    row_count: int = Field(..., description="Nombre de lignes")
    column_count: int = Field(..., description="Nombre de colonnes")
    columns: List[str] = Field(..., description="Noms des colonnes")
    file_content_base64: str = Field(..., description="Contenu du fichier encodé en base64")
    content_type: str = Field(..., description="Type MIME du contenu")
