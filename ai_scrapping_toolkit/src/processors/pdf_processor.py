"""
Module pour extraire et traiter le contenu des fichiers PDF.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import tempfile

# Configuration du logger
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrait le texte d'un fichier PDF.
    
    Args:
        pdf_path: Chemin du fichier PDF
        
    Returns:
        str: Contenu textuel du PDF
    """
    if not os.path.exists(pdf_path):
        logger.error(f"Le fichier PDF n'existe pas: {pdf_path}")
        return ""
    
    # Essayer d'abord avec PyMuPDF (fitz) qui est généralement meilleur
    try:
        import fitz  # PyMuPDF
        logger.info(f"Extraction du texte avec PyMuPDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extraire le texte de chaque page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
            text += "\n\n"  # Séparateur entre les pages
        
        doc.close()
        return text
    
    except ImportError:
        logger.warning("PyMuPDF non disponible, utilisation de PyPDF2 comme solution de repli")
    except Exception as e:
        logger.warning(f"Erreur avec PyMuPDF: {str(e)}, essai avec PyPDF2")
    
    # Solution de repli avec PyPDF2
    try:
        import PyPDF2
        logger.info(f"Extraction du texte avec PyPDF2: {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extraire le texte de chaque page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
                text += "\n\n"  # Séparateur entre les pages
            
            return text
    
    except ImportError:
        logger.error("Ni PyMuPDF ni PyPDF2 ne sont disponibles. Impossible d'extraire le texte du PDF.")
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte PDF avec PyPDF2: {str(e)}")
    
    # Si tout échoue, essayer avec pdf2image + OCR si disponible
    try:
        import pdf2image
        import pytesseract
        from PIL import Image
        
        logger.info(f"Extraction du texte avec pdf2image + OCR: {pdf_path}")
        
        with tempfile.TemporaryDirectory() as path:
            images = pdf2image.convert_from_path(pdf_path, output_folder=path)
            text = ""
            
            for i, image in enumerate(images):
                text += pytesseract.image_to_string(image)
                text += "\n\n"
            
            return text
    
    except ImportError:
        logger.error("pdf2image et/ou pytesseract ne sont pas disponibles.")
    except Exception as e:
        logger.error(f"Erreur lors de l'OCR du PDF: {str(e)}")
    
    return ""

def extract_images_from_pdf(pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Extrait les images d'un fichier PDF.
    
    Args:
        pdf_path: Chemin du fichier PDF
        output_dir: Répertoire pour enregistrer les images extraites
        
    Returns:
        List[str]: Liste des chemins des images extraites
    """
    try:
        import fitz  # PyMuPDF
        
        # Créer un répertoire temporaire si nécessaire
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix="pdf_images_")
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Extraction des images du PDF vers {output_dir}")
        
        # Ouvrir le document
        document = fitz.open(pdf_path)
        image_paths = []
        
        # Parcourir chaque page
        for page_num in range(len(document)):
            page = document[page_num]
            image_list = page.get_images(full=True)
            
            # Extraire les images de la page
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Enregistrer l'image
                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                image_paths.append(image_path)
        
        document.close()
        logger.info(f"Extraction d'images terminée: {len(image_paths)} images extraites")
        return image_paths
    
    except ImportError:
        logger.error("PyMuPDF est nécessaire pour extraire les images. Installez-le avec 'pip install pymupdf'.")
        return []
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des images: {e}")
        return []

def pdf_to_chunks(
    pdf_path: str, 
    method: str = 'pages', 
    max_length: int = 1000, 
    overlap: int = 100
) -> List[str]:
    """
    Convertit un fichier PDF en une liste de chunks de texte.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        method: Méthode de découpage ('pages', 'paragraphs', 'length')
        max_length: Taille maximale d'un chunk en caractères
        overlap: Chevauchement entre chunks consécutifs
        
    Returns:
        List[str]: Liste des chunks de texte extraits
    """
    # Extraire le texte complet du PDF
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        logger.error(f"Aucun texte extrait du PDF: {pdf_path}")
        return []
    
    # Découper selon la méthode spécifiée
    if method == 'pages':
        # Découper par pages en utilisant les sauts de page comme délimiteurs
        chunks = [page.strip() for page in text.split("\n\n") if page.strip()]
        
        # Si des chunks sont trop longs, les redécouper
        processed_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                # Redécouper les pages trop longues
                processed_chunks.extend(chunk_by_length(chunk, max_length, overlap))
            else:
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    elif method == 'paragraphs':
        # Découper par paragraphes
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        
        # Regrouper les paragraphes courts
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 1 <= max_length:
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:  # Ajouter le dernier chunk
            chunks.append(current_chunk)
        
        return chunks
    
    else:  # method == 'length' ou autre
        # Découper le texte par longueur
        return chunk_by_length(text, max_length, overlap)

def chunk_by_length(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    Découpe un texte en segments de taille maximum spécifiée avec chevauchement.
    
    Args:
        text: Texte à découper
        max_length: Longueur maximale d'un chunk
        overlap: Chevauchement entre chunks consécutifs
        
    Returns:
        List[str]: Liste des chunks
    """
    if not text:
        return []
    
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Déterminer la fin du chunk actuel
        end = start + max_length
        
        if end >= len(text):
            # Dernier chunk
            chunks.append(text[start:])
            break
        
        # Chercher la dernière occurrence d'un caractère de séparation
        separators = ['. ', '? ', '! ', '\n\n', '\n', '. ', ', ', ' ']
        chunk_end = end
        
        for sep in separators:
            last_sep_pos = text[start:end].rfind(sep)
            if last_sep_pos != -1:
                chunk_end = start + last_sep_pos + len(sep)
                break
        
        # Si aucun séparateur n'est trouvé, découper au dernier espace
        if chunk_end == end:
            last_space = text[start:end].rfind(' ')
            if last_space != -1:
                chunk_end = start + last_space + 1
        
        # Si toujours pas de bon point de coupure, couper à la position max
        if chunk_end == end and end < len(text):
            chunk_end = end
        
        # Ajouter le chunk
        chunks.append(text[start:chunk_end])
        
        # Mettre à jour la position de départ avec chevauchement
        start = chunk_end - overlap if chunk_end > start + overlap else chunk_end
    
    return chunks

def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extrait les métadonnées d'un fichier PDF.
    
    Args:
        pdf_path: Chemin du fichier PDF
        
    Returns:
        Dict[str, Any]: Métadonnées du PDF
    """
    try:
        import fitz  # PyMuPDF
        document = fitz.open(pdf_path)
        metadata = document.metadata
        
        # Ajouter des informations supplémentaires
        metadata['page_count'] = len(document)
        metadata['file_size_bytes'] = os.path.getsize(pdf_path)
        
        document.close()
        return metadata
    
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            
            metadata = {}
            if reader.metadata:
                for key, value in reader.metadata.items():
                    # Convert internal PDF key names to readable names
                    clean_key = key.strip('/').lower()
                    metadata[clean_key] = value
            
            metadata['page_count'] = len(reader.pages)
            metadata['file_size_bytes'] = os.path.getsize(pdf_path)
            
            return metadata
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des métadonnées avec PyPDF2: {e}")
            return {'error': str(e)}
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des métadonnées: {e}")
        return {'error': str(e)}
