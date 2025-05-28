"""
Module de chunking sémantique qui préserve la cohérence des informations
et priorise les sections importantes basées sur la structure HTML.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SemanticChunk:
    """Représente un chunk avec ses métadonnées sémantiques."""
    content: str
    priority: int  # 1 = très importante, 2 = importante, 3 = normale
    section_type: str  # 'specs', 'review', 'pricing', 'general'
    html_tags: List[str]  # Tags HTML présents dans ce chunk
    keywords: Set[str]  # Mots-clés détectés
    start_position: int  # Position dans le document original
    end_position: int

class SemanticChunker:
    """Chunker sémantique basé sur la structure HTML et le contenu."""
    
    def __init__(self):
        # Définir les patterns pour identifier les types de sections
        self.section_patterns = {
            'specs': {
                'keywords': [
                    'spécifications', 'caractéristiques', 'fiche technique', 'specs',
                    'processeur', 'cpu', 'ram', 'mémoire', 'stockage', 'écran', 'display',
                    'batterie', 'autonomie', 'capteur', 'appareil photo', 'camera',
                    'dimensions', 'poids', 'connectivité', 'bluetooth', 'wifi', 'nfc'
                ],
                'html_classes': ['specs', 'specifications', 'technical', 'features'],
                'html_ids': ['specs', 'specifications', 'technical-specs'],
                'priority': 1
            },
            'review': {
                'keywords': [
                    'test', 'avis', 'opinion', 'critique', 'évaluation', 'verdict',
                    'points forts', 'points faibles', 'avantages', 'inconvénients',
                    'performance', 'qualité', 'note', 'rating', 'score'
                ],
                'html_classes': ['review', 'opinion', 'verdict', 'pros-cons'],
                'html_ids': ['review', 'verdict', 'conclusion'],
                'priority': 1
            },
            'pricing': {
                'keywords': [
                    'prix', 'tarif', 'coût', 'euro', '€', '$', 'dollar',
                    'promotion', 'offre', 'réduction', 'disponibilité',
                    'achat', 'commander', 'boutique', 'magasin'
                ],
                'html_classes': ['price', 'pricing', 'offer', 'shop'],
                'html_ids': ['price', 'pricing', 'buy'],
                'priority': 2
            },
            'general': {
                'keywords': [],
                'html_classes': [],
                'html_ids': [],
                'priority': 3
            }
        }
    
    def semantic_chunk_html(
        self, 
        html_content: str, 
        max_chunk_size: int = 4000,
        min_chunk_size: int = 200,
        overlap: int = 100
    ) -> List[SemanticChunk]:
        """
        Découpe le contenu HTML en chunks sémantiques.
        
        Args:
            html_content: Contenu HTML à analyser
            max_chunk_size: Taille maximale d'un chunk
            min_chunk_size: Taille minimale d'un chunk
            overlap: Chevauchement entre chunks
            
        Returns:
            Liste de chunks sémantiques triés par priorité
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Identifier les sections importantes
            important_sections = self._identify_important_sections(soup)
            
            # Créer des chunks à partir des sections importantes
            semantic_chunks = []
            
            for section in important_sections:
                chunks = self._create_chunks_from_section(
                    section, max_chunk_size, min_chunk_size, overlap
                )
                semantic_chunks.extend(chunks)
            
            # Traiter le contenu restant
            remaining_content = self._get_remaining_content(soup, important_sections)
            if remaining_content:
                general_chunks = self._create_general_chunks(
                    remaining_content, max_chunk_size, min_chunk_size, overlap
                )
                semantic_chunks.extend(general_chunks)
            
            # Trier par priorité et position
            semantic_chunks.sort(key=lambda x: (x.priority, x.start_position))
            
            return semantic_chunks
            
        except Exception as e:
            logger.error(f"Erreur lors du chunking sémantique: {e}")
            # Fallback vers le chunking traditionnel
            return self._fallback_chunking(html_content, max_chunk_size, overlap)
    
    def _identify_important_sections(self, soup: BeautifulSoup) -> List[Dict]:
        """Identifie les sections importantes dans le HTML."""
        important_sections = []
        
        # Rechercher par structure HTML (h1, h2, h3, sections, articles)
        structural_elements = soup.find_all(['h1', 'h2', 'h3', 'section', 'article', 'div'])
        
        for element in structural_elements:
            section_info = self._analyze_section(element)
            if section_info['importance'] > 0:
                important_sections.append(section_info)
        
        # Rechercher par tableaux (souvent utilisés pour les specs)
        tables = soup.find_all('table')
        for table in tables:
            table_info = self._analyze_table(table)
            if table_info['importance'] > 0:
                important_sections.append(table_info)
        
        return important_sections
    
    def _analyze_section(self, element: Tag) -> Dict:
        """Analyse un élément HTML pour déterminer son importance."""
        section_info = {
            'element': element,
            'type': 'general',
            'importance': 0,
            'keywords': set(),
            'html_tags': []
        }
        
        # Obtenir le texte de l'élément
        text_content = element.get_text().lower()
        
        # Analyser les attributs HTML
        classes = element.get('class', [])
        element_id = element.get('id', '')
        
        # Vérifier chaque type de section
        for section_type, patterns in self.section_patterns.items():
            if section_type == 'general':
                continue
                
            importance_score = 0
            matched_keywords = set()
            
            # Vérifier les mots-clés
            for keyword in patterns['keywords']:
                if keyword in text_content:
                    importance_score += 1
                    matched_keywords.add(keyword)
            
            # Vérifier les classes HTML
            for html_class in patterns['html_classes']:
                if any(html_class in cls.lower() for cls in classes):
                    importance_score += 2
            
            # Vérifier l'ID HTML
            for html_id in patterns['html_ids']:
                if html_id in element_id.lower():
                    importance_score += 3
            
            # Bonus pour les titres qui contiennent des mots-clés pertinents
            if element.name in ['h1', 'h2', 'h3'] and matched_keywords:
                importance_score += 2
            
            if importance_score > section_info['importance']:
                section_info.update({
                    'type': section_type,
                    'importance': importance_score,
                    'keywords': matched_keywords,
                    'html_tags': [element.name] + classes
                })
        
        return section_info
    
    def _analyze_table(self, table: Tag) -> Dict:
        """Analyse un tableau pour détecter s'il contient des spécifications."""
        table_text = table.get_text().lower()
        specs_keywords = self.section_patterns['specs']['keywords']
        
        keyword_count = sum(1 for keyword in specs_keywords if keyword in table_text)
        
        return {
            'element': table,
            'type': 'specs' if keyword_count >= 3 else 'general',
            'importance': keyword_count,
            'keywords': {kw for kw in specs_keywords if kw in table_text},
            'html_tags': ['table']
        }
    
    def _create_chunks_from_section(
        self, 
        section_info: Dict, 
        max_size: int, 
        min_size: int, 
        overlap: int
    ) -> List[SemanticChunk]:
        """Crée des chunks à partir d'une section identifiée."""
        element = section_info['element']
        text_content = element.get_text(separator='\n', strip=True)
        
        if len(text_content) < min_size:
            # Section trop petite, essayer de l'étendre avec les éléments suivants
            text_content = self._extend_small_section(element, min_size)
        
        chunks = []
        if len(text_content) <= max_size:
            # Section tient dans un seul chunk
            chunk = SemanticChunk(
                content=text_content,
                priority=self.section_patterns[section_info['type']]['priority'],
                section_type=section_info['type'],
                html_tags=section_info['html_tags'],
                keywords=section_info['keywords'],
                start_position=0,
                end_position=len(text_content)
            )
            chunks.append(chunk)
        else:
            # Diviser la section en plusieurs chunks
            sub_chunks = self._split_large_section(text_content, max_size, overlap)
            for i, sub_chunk in enumerate(sub_chunks):
                chunk = SemanticChunk(
                    content=sub_chunk,
                    priority=self.section_patterns[section_info['type']]['priority'],
                    section_type=section_info['type'],
                    html_tags=section_info['html_tags'],
                    keywords=section_info['keywords'],
                    start_position=i * (max_size - overlap),
                    end_position=i * (max_size - overlap) + len(sub_chunk)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _extend_small_section(self, element: Tag, min_size: int) -> str:
        """Étend une petite section avec les éléments suivants."""
        content = element.get_text(separator='\n', strip=True)
        
        # Chercher les éléments frères suivants
        next_sibling = element.find_next_sibling()
        while next_sibling and len(content) < min_size:
            if isinstance(next_sibling, Tag):
                additional_text = next_sibling.get_text(separator='\n', strip=True)
                content += "\n" + additional_text
            next_sibling = next_sibling.find_next_sibling()
        
        return content
    
    def _split_large_section(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Divise intelligemment une grande section en chunks."""
        chunks = []
        
        # Essayer de diviser par paragraphes d'abord
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Si le paragraphe est trop long, le diviser
                if len(paragraph) > max_size:
                    sub_chunks = self._split_by_sentences(paragraph, max_size, overlap)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Divise un texte par phrases."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_remaining_content(self, soup: BeautifulSoup, processed_sections: List[Dict]) -> str:
        """Récupère le contenu non traité."""
        # Pour simplifier, on retourne le texte principal moins les sections déjà traitées
        # Cette implémentation peut être améliorée pour être plus précise
        all_text = soup.get_text(separator='\n', strip=True)
        
        # Enlever le contenu des sections déjà traitées
        for section_info in processed_sections:
            section_text = section_info['element'].get_text(separator='\n', strip=True)
            all_text = all_text.replace(section_text, '', 1)
        
        return all_text.strip()
    
    def _create_general_chunks(
        self, 
        content: str, 
        max_size: int, 
        min_size: int, 
        overlap: int
    ) -> List[SemanticChunk]:
        """Crée des chunks généraux à partir du contenu restant."""
        if len(content) < min_size:
            return []
        
        chunks = []
        text_chunks = self._split_by_sentences(content, max_size, overlap)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = SemanticChunk(
                content=chunk_text,
                priority=3,  # Priorité générale
                section_type='general',
                html_tags=[],
                keywords=set(),
                start_position=i * (max_size - overlap),
                end_position=i * (max_size - overlap) + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_chunking(self, html_content: str, max_size: int, overlap: int) -> List[SemanticChunk]:
        """Chunking de secours en cas d'erreur."""
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        
        chunks = []
        text_chunks = self._split_by_sentences(text, max_size, overlap)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = SemanticChunk(
                content=chunk_text,
                priority=3,
                section_type='general',
                html_tags=[],
                keywords=set(),
                start_position=i * (max_size - overlap),
                end_position=i * (max_size - overlap) + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks


def semantic_html_to_chunks(
    html_content: str,
    max_length: int = 4000,
    min_length: int = 200,
    overlap: int = 100,
    prioritize_important: bool = True
) -> List[str]:
    """
    Interface publique pour le chunking sémantique.
    
    Args:
        html_content: Contenu HTML à analyser
        max_length: Taille maximale d'un chunk
        min_length: Taille minimale d'un chunk
        overlap: Chevauchement entre chunks
        prioritize_important: Si True, priorise les sections importantes
        
    Returns:
        Liste de chunks ordonnés par importance
    """
    chunker = SemanticChunker()
    semantic_chunks = chunker.semantic_chunk_html(
        html_content, max_length, min_length, overlap
    )
    
    if prioritize_important:
        # Trier par priorité puis par position
        semantic_chunks.sort(key=lambda x: (x.priority, x.start_position))
    else:
        # Trier seulement par position (ordre naturel)
        semantic_chunks.sort(key=lambda x: x.start_position)
    
    # Retourner seulement le contenu texte
    return [chunk.content for chunk in semantic_chunks]
