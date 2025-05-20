"""
Modèles Pydantic pour les requêtes API.
"""

from pydantic import BaseModel, Field, HttpUrl, validator, AnyHttpUrl
from typing import Optional, List, Dict, Any, Union, Literal


class ScrapingRequest(BaseModel):
    """Modèle pour les requêtes de scraping."""
    url: HttpUrl = Field(..., description="URL du site à scraper")
    method: Literal["requests", "selenium", "auto"] = Field(
        "auto", description="Méthode de scraping à utiliser"
    )
    wait_time: int = Field(5, description="Temps d'attente pour Selenium (secondes)")
    preprocess: bool = Field(False, description="Prétraiter le HTML")
    extract_main_content: bool = Field(
        False, description="Extraire uniquement le contenu principal"
    )
    respect_robots: bool = Field(
        True, description="Respecter les règles robots.txt"
    )
    user_agent: Optional[str] = Field(
        None, description="User-Agent à utiliser pour les requêtes"
    )
    rate_limit: float = Field(
        1.0, description="Délai minimum entre les requêtes (secondes)"
    )


class ChunkingRequest(BaseModel):
    """Modèle pour les requêtes de chunking."""
    content: str = Field(..., description="Contenu HTML à découper en chunks")
    chunk_method: Literal["tags", "length", "hybrid"] = Field(
        "hybrid", description="Méthode de découpage"
    )
    max_length: int = Field(1000, description="Taille maximale d'un chunk")
    overlap: int = Field(100, description="Chevauchement entre les chunks")


class VectorizationRequest(BaseModel):
    """Modèle pour les requêtes de vectorisation."""
    chunks: List[str] = Field(..., description="Liste de chunks à vectoriser")
    model_name: str = Field(
        "all-MiniLM-L6-v2", description="Modèle sentence-transformers à utiliser"
    )
    index_type: Literal["L2", "IP", "IVF"] = Field(
        "L2", description="Type d'index FAISS à créer"
    )


class SearchRequest(BaseModel):
    """Modèle pour les requêtes de recherche."""
    query: str = Field(..., description="Requête de recherche")
    index_path: str = Field(..., description="Chemin de l'index vectoriel")
    top_k: int = Field(5, description="Nombre de résultats à retourner")
    model_name: str = Field(
        "all-MiniLM-L6-v2", description="Modèle sentence-transformers à utiliser"
    )


class ExtractionRequest(BaseModel):
    """Modèle pour les requêtes d'extraction de données."""
    content: str = Field(..., description="Contenu HTML à analyser")
    query: str = Field(
        ..., description="Requête d'extraction (ex: 'Extraire tous les titres et dates')"
    )
    provider: Literal["openai", "ollama", "lmstudio", "huggingface", "openrouter"] = Field(
        "openai", description="Provider du modèle de langage"
    )
    model: str = Field("gpt-3.5-turbo", description="Modèle à utiliser")
    chunk_size: int = Field(4000, description="Taille maximale des chunks")
    max_chunks: Optional[int] = Field(None, description="Nombre maximum de chunks à traiter")
    chunk_method: Literal["tags", "length", "hybrid"] = Field(
        "hybrid", description="Méthode de chunking"
    )
    temperature: float = Field(0.0, description="Température pour la génération")
    api_key: Optional[str] = Field(None, description="Clé API pour le provider")
    host: Optional[str] = Field(None, description="URL du serveur API (pour lmstudio et ollama)")


class PDFExtractionRequest(BaseModel):
    """Modèle pour les requêtes d'extraction de données depuis un PDF."""
    query: str = Field(..., description="Requête d'extraction en langage naturel")
    provider: Literal["openai", "ollama", "lmstudio", "huggingface", "openrouter"] = Field(
        "openai", description="Provider du modèle de langage"
    )
    model: str = Field("gpt-3.5-turbo", description="Modèle à utiliser")
    chunk_method: Literal["pages", "paragraphs", "length"] = Field(
        "pages", description="Méthode de chunking pour PDF"
    )
    chunk_size: int = Field(4000, description="Taille maximale des chunks")
    max_chunks: Optional[int] = Field(None, description="Nombre maximum de chunks à traiter")
    temperature: float = Field(0.0, description="Température pour la génération")
    api_key: Optional[str] = Field(None, description="Clé API pour le provider")
    auto_enhance: bool = Field(True, description="Améliorer automatiquement les requêtes génériques")


class ProcessingRequest(BaseModel):
    """Modèle pour les requêtes de traitement de données."""
    data: Dict[str, Any] = Field(..., description="Données à traiter")
    filter_date: bool = Field(False, description="Filtrer par date")
    date_field: Optional[str] = Field(None, description="Nom du champ date")
    days: int = Field(30, description="Nombre de jours à considérer")
    start_date: Optional[str] = Field(None, description="Date de début")
    end_date: Optional[str] = Field(None, description="Date de fin")
    analyze_sentiment: bool = Field(False, description="Analyser le sentiment")
    sentiment_field: Optional[str] = Field(None, description="Champ pour le sentiment")
    sentiment_provider: Optional[str] = Field(None, description="Provider pour le sentiment")
    categorize: bool = Field(False, description="Catégoriser les textes")
    category_field: Optional[str] = Field(None, description="Champ pour la catégorisation")
    categories: Optional[List[str]] = Field(None, description="Liste des catégories")
    sort_by: Optional[str] = Field(None, description="Champ pour le tri")
    sort_desc: bool = Field(False, description="Tri par ordre décroissant")
    filter_expr: Optional[str] = Field(None, description="Expression de filtrage")


class ExportRequest(BaseModel):
    """Modèle pour les requêtes d'export de données."""
    data: Dict[str, Any] = Field(..., description="Données à exporter")
    output_format: Literal["csv", "json", "excel"] = Field(
        "csv", description="Format de sortie"
    )
    remove_duplicates: bool = Field(True, description="Supprimer les doublons")
    date_columns: Optional[List[str]] = Field(None, description="Colonnes de date")
    date_format: str = Field("%Y-%m-%d", description="Format des dates")
    sort_by: Optional[str] = Field(None, description="Colonne pour le tri")
    sort_ascending: bool = Field(True, description="Tri ascendant")
    columns: Optional[List[str]] = Field(None, description="Colonnes à inclure")
    include_index: bool = Field(False, description="Inclure l'index")
    flatten_complex: bool = Field(True, description="Aplatir les structures complexes")
