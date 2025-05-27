"""
Module d'extraction de données amélioré avec traitement en deux passes et agrégation complète.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class WebsiteType(Enum):
    """Types de sites web pour adapter les prompts."""
    REVIEW = "review"
    ECOMMERCE = "ecommerce"
    NEWS = "news"
    BLOG = "blog"
    TECHNICAL = "technical"
    GENERAL = "general"

@dataclass
class ChunkAnalysis:
    """Résultat de l'analyse d'un chunk."""
    chunk_index: int
    content_type: str
    relevance_score: float
    contains_specs: bool
    contains_pricing: bool
    contains_review: bool
    key_topics: List[str]
    priority: int

class EnhancedDataExtractor:
    """Extracteur de données amélioré avec traitement en deux passes."""
    
    def __init__(self):
        self.website_patterns = {
            WebsiteType.REVIEW: {
                'domains': ['frandroid.com', 'phonandroid.com', 'numerama.com', 'clubic.com'],
                'indicators': ['test', 'review', 'avis', 'critique', 'verdict', 'note', 'rating'],
                'priority_sections': ['specs', 'pros', 'cons', 'rating', 'conclusion']
            },
            WebsiteType.ECOMMERCE: {
                'domains': ['cdiscount.com', 'amazon.fr', 'fnac.com', 'darty.com'],
                'indicators': ['prix', 'acheter', 'ajouter panier', 'promotion', 'offre'],
                'priority_sections': ['price', 'product', 'offer', 'specs', 'availability']
            },
            WebsiteType.NEWS: {
                'domains': ['lemonde.fr', 'lefigaro.fr', 'bfmtv.com'],
                'indicators': ['actualité', 'news', 'info', 'article'],
                'priority_sections': ['headline', 'summary', 'content', 'details']
            }
        }
    
    def detect_website_type(self, url: str, content: str) -> WebsiteType:
        """Détecte le type de site web basé sur l'URL et le contenu."""
        domain = self._extract_domain(url)
        content_lower = content.lower()
        
        # Vérifier les domaines connus
        for website_type, patterns in self.website_patterns.items():
            if any(domain in patterns['domains'] for domain in patterns['domains']):
                return website_type
        
        # Analyser le contenu pour déterminer le type
        review_score = sum(1 for indicator in self.website_patterns[WebsiteType.REVIEW]['indicators'] 
                          if indicator in content_lower)
        ecommerce_score = sum(1 for indicator in self.website_patterns[WebsiteType.ECOMMERCE]['indicators'] 
                             if indicator in content_lower)
        
        if review_score > ecommerce_score and review_score >= 2:
            return WebsiteType.REVIEW
        elif ecommerce_score >= 2:
            return WebsiteType.ECOMMERCE
        
        return WebsiteType.GENERAL
    
    def _extract_domain(self, url: str) -> str:
        """Extrait le domaine d'une URL."""
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.lower()
    
    def analyze_chunks_first_pass(
        self, 
        chunks: List[str], 
        website_type: WebsiteType,
        llm_provider,
        query: str
    ) -> List[ChunkAnalysis]:
        """
        Première passe : analyser chaque chunk pour identifier son contenu.
        """
        logger.info(f"Première passe : analyse de {len(chunks)} chunks pour type {website_type.value}")
        
        analysis_prompt = self._create_analysis_prompt(website_type, query)
        
        analyses = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {
                executor.submit(self._analyze_single_chunk, chunk, i, analysis_prompt, llm_provider): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_index):
                chunk_index = future_to_index[future]
                try:
                    analysis = future.result()
                    analysis.chunk_index = chunk_index
                    analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Erreur lors de l'analyse du chunk {chunk_index}: {e}")
                    # Créer une analyse par défaut
                    default_analysis = ChunkAnalysis(
                        chunk_index=chunk_index,
                        content_type="unknown",
                        relevance_score=0.5,
                        contains_specs=False,
                        contains_pricing=False,
                        contains_review=False,
                        key_topics=[],
                        priority=3
                    )
                    analyses.append(default_analysis)
        
        # Trier par index pour maintenir l'ordre
        analyses.sort(key=lambda x: x.chunk_index)
        
        return analyses
    
    def _create_analysis_prompt(self, website_type: WebsiteType, original_query: str) -> str:
        """Crée un prompt pour l'analyse des chunks."""
        base_prompt = f"""Analyse ce fragment de texte et détermine son contenu. Réponds uniquement avec un objet JSON valide.

Requête originale de l'utilisateur: {original_query}

Structure JSON attendue:
{{
  "content_type": "specs|pricing|review|product_list|general",
  "relevance_score": 0.0-1.0,
  "contains_specs": true/false,
  "contains_pricing": true/false,
  "contains_review": true/false,
  "key_topics": ["topic1", "topic2"],
  "priority": 1-3
}}

"""
        
        if website_type == WebsiteType.REVIEW:
            base_prompt += """
Pour un site de test/review, identifie particulièrement:
- Spécifications techniques (specs)
- Points forts/faibles (review)
- Notes/scores (review)
- Prix mentionnés (pricing)
"""
        elif website_type == WebsiteType.ECOMMERCE:
            base_prompt += """
Pour un site e-commerce, identifie particulièrement:
- Listes de produits (product_list)
- Prix et promotions (pricing)
- Caractéristiques produits (specs)
- Disponibilité (general)
"""
        
        return base_prompt
    
    def _analyze_single_chunk(
        self, 
        chunk: str, 
        index: int, 
        prompt: str, 
        llm_provider
    ) -> ChunkAnalysis:
        """Analyse un chunk individuel."""
        try:
            full_prompt = f"{prompt}\n\nTexte à analyser:\n{chunk[:2000]}..."
            
            result = llm_provider.extract(chunk, full_prompt, "json")
            
            if isinstance(result, dict) and 'content_type' in result:
                return ChunkAnalysis(
                    chunk_index=index,
                    content_type=result.get('content_type', 'general'),
                    relevance_score=float(result.get('relevance_score', 0.5)),
                    contains_specs=result.get('contains_specs', False),
                    contains_pricing=result.get('contains_pricing', False),
                    contains_review=result.get('contains_review', False),
                    key_topics=result.get('key_topics', []),
                    priority=int(result.get('priority', 3))
                )
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse du chunk {index}: {e}")
        
        # Analyse heuristique en cas d'échec
        return self._heuristic_analysis(chunk, index)
    
    def _heuristic_analysis(self, chunk: str, index: int) -> ChunkAnalysis:
        """Analyse heuristique d'un chunk en cas d'échec du LLM."""
        chunk_lower = chunk.lower()
        
        # Détecter les spécifications
        specs_keywords = ['processeur', 'cpu', 'ram', 'mémoire', 'stockage', 'écran', 'batterie', 'mah', 'mp', 'ghz']
        contains_specs = sum(1 for kw in specs_keywords if kw in chunk_lower) >= 2
        
        # Détecter les prix
        pricing_keywords = ['€', 'prix', 'coût', 'promotion', 'offre', 'tarif']
        contains_pricing = any(kw in chunk_lower for kw in pricing_keywords) or bool(re.search(r'\d+\s*€', chunk))
        
        # Détecter les reviews
        review_keywords = ['points forts', 'points faibles', 'verdict', 'note', 'avis', 'critique']
        contains_review = sum(1 for kw in review_keywords if kw in chunk_lower) >= 1
        
        # Déterminer le type de contenu
        if contains_specs and len([kw for kw in specs_keywords if kw in chunk_lower]) >= 3:
            content_type = "specs"
            priority = 1
        elif contains_pricing:
            content_type = "pricing"
            priority = 2
        elif contains_review:
            content_type = "review"
            priority = 1
        else:
            content_type = "general"
            priority = 3
        
        # Score de pertinence basé sur la longueur et les mots-clés
        relevance_score = min(1.0, len(chunk) / 1000 + (contains_specs * 0.3) + (contains_pricing * 0.2) + (contains_review * 0.3))
        
        return ChunkAnalysis(
            chunk_index=index,
            content_type=content_type,
            relevance_score=relevance_score,
            contains_specs=contains_specs,
            contains_pricing=contains_pricing,
            contains_review=contains_review,
            key_topics=[],
            priority=priority
        )
    
    def extract_from_prioritized_chunks(
        self,
        chunks: List[str],
        analyses: List[ChunkAnalysis],
        website_type: WebsiteType,
        original_query: str,
        llm_provider,
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Deuxième passe : extraction ciblée des chunks les plus pertinents.
        """
        # Trier les chunks par priorité et score de pertinence
        sorted_analyses = sorted(analyses, key=lambda x: (x.priority, -x.relevance_score))
        
        if max_chunks:
            sorted_analyses = sorted_analyses[:max_chunks]
        
        logger.info(f"Deuxième passe : extraction de {len(sorted_analyses)} chunks prioritaires")
        
        # Créer des prompts spécialisés par type de contenu
        extraction_results = {}
        
        # Grouper les chunks par type de contenu
        chunks_by_type = {}
        for analysis in sorted_analyses:
            content_type = analysis.content_type
            if content_type not in chunks_by_type:
                chunks_by_type[content_type] = []
            chunks_by_type[content_type].append((chunks[analysis.chunk_index], analysis))
        
        # Traiter chaque type de contenu
        for content_type, chunk_list in chunks_by_type.items():
            specialized_prompt = self._create_specialized_extraction_prompt(
                content_type, website_type, original_query
            )
            
            type_results = []
            for chunk_content, analysis in chunk_list:
                try:
                    result = llm_provider.extract(chunk_content, specialized_prompt, "json")
                    if isinstance(result, dict):
                        type_results.append(result)
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction du chunk {analysis.chunk_index}: {e}")
            
            if type_results:
                extraction_results[content_type] = type_results
        
        # Agréger tous les résultats
        return self._aggregate_extraction_results(extraction_results, website_type, original_query)
    
    def _create_specialized_extraction_prompt(
        self, 
        content_type: str, 
        website_type: WebsiteType, 
        original_query: str
    ) -> str:
        """Crée un prompt spécialisé pour un type de contenu."""
        base_instruction = f"Requête utilisateur: {original_query}\n\n"
        
        if content_type == "specs" and website_type == WebsiteType.REVIEW:
            return base_instruction + """Extrait uniquement les spécifications techniques de ce texte.
Structure JSON attendue:
{
  "ecran": {"taille": "", "resolution": "", "technologie": ""},
  "processeur": {"modele": "", "frequence": ""},
  "memoire": {"ram": "", "stockage": ""},
  "photo": {"principal": "", "ultra_grand_angle": "", "zoom": ""},
  "batterie": {"capacite": "", "charge_rapide": ""},
  "autres": {"os": "", "connectivite": []}
}"""
        
        elif content_type == "review" and website_type == WebsiteType.REVIEW:
            return base_instruction + """Extrait les éléments d'évaluation de ce texte.
Structure JSON attendue:
{
  "points_forts": ["point 1", "point 2"],
  "points_faibles": ["point 1", "point 2"],
  "note_globale": "",
  "verdict": ""
}"""
        
        elif content_type == "pricing":
            return base_instruction + """Extrait les informations de prix de ce texte.
Structure JSON attendue:
{
  "prix_principal": "",
  "prix_min": "",
  "prix_max": "",
  "promotions": ["promo 1", "promo 2"],
  "disponibilite": ""
}"""
        
        elif content_type == "product_list" and website_type == WebsiteType.ECOMMERCE:
            return base_instruction + """Extrait la liste des produits de ce texte.
Structure JSON attendue:
{
  "produits": [
    {"nom": "", "prix": "", "description": ""},
    {"nom": "", "prix": "", "description": ""}
  ]
}"""
        
        else:
            return base_instruction + f"""Extrait les informations pertinentes de ce texte de type '{content_type}'.
Réponds avec un objet JSON structuré contenant les informations importantes."""
    
    def _aggregate_extraction_results(
        self, 
        extraction_results: Dict[str, List[Dict]], 
        website_type: WebsiteType,
        original_query: str
    ) -> Dict[str, Any]:
        """Agrège les résultats d'extraction de tous les chunks."""
        aggregated = {}
        
        # Agréger les spécifications techniques
        if "specs" in extraction_results:
            specs = {}
            for result in extraction_results["specs"]:
                for key, value in result.items():
                    if key not in specs:
                        specs[key] = value
                    elif isinstance(value, dict) and isinstance(specs[key], dict):
                        specs[key].update(value)
            aggregated["caracteristiques_techniques"] = specs
        
        # Agréger les évaluations
        if "review" in extraction_results:
            points_forts = []
            points_faibles = []
            notes = []
            verdicts = []
            
            for result in extraction_results["review"]:
                if "points_forts" in result:
                    points_forts.extend(result["points_forts"])
                if "points_faibles" in result:
                    points_faibles.extend(result["points_faibles"])
                if "note_globale" in result:
                    notes.append(result["note_globale"])
                if "verdict" in result:
                    verdicts.append(result["verdict"])
            
            aggregated.update({
                "points_forts": list(set(points_forts)),  # Supprimer les doublons
                "points_faibles": list(set(points_faibles)),
                "note_globale": notes[0] if notes else None,
                "verdict": " ".join(verdicts) if verdicts else None
            })
        
        # Agréger les prix
        if "pricing" in extraction_results:
            prix_info = {"prix": [], "promotions": []}
            for result in extraction_results["pricing"]:
                if "prix_principal" in result and result["prix_principal"]:
                    prix_info["prix"].append(result["prix_principal"])
                if "promotions" in result:
                    prix_info["promotions"].extend(result["promotions"])
            aggregated["prix_et_disponibilite"] = prix_info
        
        # Agréger les listes de produits
        if "product_list" in extraction_results:
            produits = []
            for result in extraction_results["product_list"]:
                if "produits" in result:
                    produits.extend(result["produits"])
            aggregated["produits"] = produits
        
        # Ajouter les informations générales
        if "general" in extraction_results:
            general_info = {}
            for result in extraction_results["general"]:
                general_info.update(result)
            if general_info:
                aggregated["informations_generales"] = general_info
        
        return aggregated

def enhanced_extract_data_from_chunks(
    chunks: List[str],
    query: str,
    llm_provider,
    url: str = "",
    max_workers: int = 4,
    max_chunks_per_type: int = 5
) -> Dict[str, Any]:
    """
    Point d'entrée principal pour l'extraction améliorée en deux passes.
    """
    extractor = EnhancedDataExtractor()
    
    # Détecter le type de site web
    content_sample = " ".join(chunks[:3])  # Échantillon des premiers chunks
    website_type = extractor.detect_website_type(url, content_sample)
    
    logger.info(f"Type de site détecté: {website_type.value}")
    
    # Première passe : analyser tous les chunks
    analyses = extractor.analyze_chunks_first_pass(chunks, website_type, llm_provider, query)
    
    # Deuxième passe : extraction ciblée
    results = extractor.extract_from_prioritized_chunks(
        chunks, analyses, website_type, query, llm_provider, max_chunks_per_type * 4
    )
    
    return results
