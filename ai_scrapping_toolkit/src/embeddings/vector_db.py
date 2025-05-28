"""
Module pour la conversion de texte en vecteurs d'embedding
et la gestion d'une base de données vectorielle avec FAISS.
"""

import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Union, Optional, Any

def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Charge un modèle de sentence-transformers pour générer des embeddings.
    
    Args:
        model_name (str): Nom du modèle à utiliser
        
    Returns:
        SentenceTransformer: Instance du modèle chargé
    """
    try:
        model = SentenceTransformer(model_name)
        print(f"Modèle d'embedding '{model_name}' chargé avec succès.")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

def chunks_to_embeddings(
    chunks: List[str], 
    model: Optional[SentenceTransformer] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Convertit une liste de chunks de texte en vecteurs d'embedding.
    
    Args:
        chunks (List[str]): Liste de chunks de texte à vectoriser
        model (SentenceTransformer, optional): Modèle préchargé, ou None pour en charger un nouveau
        model_name (str): Nom du modèle à utiliser si model=None
        batch_size (int): Nombre de chunks à traiter par lot
        show_progress (bool): Afficher la progression
        
    Returns:
        np.ndarray: Matrice des vecteurs d'embedding
    """
    if not chunks:
        raise ValueError("La liste de chunks est vide")
    
    # Charger le modèle si non fourni
    if model is None:
        model = load_embedding_model(model_name)
    
    # Générer les embeddings
    try:
        embeddings = model.encode(
            chunks, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # Normalisation pour la similarité cosinus
        )
        print(f"Embeddings générés avec succès: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Erreur lors de la génération des embeddings: {str(e)}")
        raise

def create_faiss_index(
    embeddings: np.ndarray,
    chunks: List[str],
    index_type: str = 'L2',
    metadata: Optional[List[Dict[str, Any]]] = None
) -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Crée un index FAISS à partir des vecteurs d'embedding.
    
    Args:
        embeddings (np.ndarray): Matrice des vecteurs d'embedding
        chunks (List[str]): Liste de chunks de texte correspondants
        index_type (str): Type d'index FAISS ('L2', 'IP', 'Flat', 'IVF', etc.)
        metadata (List[Dict]): Metadata optionnelle pour chaque chunk
        
    Returns:
        Tuple[faiss.Index, Dict]: Index FAISS et dictionnaire de métadonnées
    """
    if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
        raise ValueError("Les embeddings fournis ne sont pas valides")
    
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"Le nombre de chunks ({len(chunks)}) ne correspond pas au nombre d'embeddings ({embeddings.shape[0]})")
    
    # Dimension des vecteurs
    dim = embeddings.shape[1]
    
    # Créer l'index FAISS selon le type spécifié
    if index_type == 'L2':
        index = faiss.IndexFlatL2(dim)  # Distance L2 (euclidienne)
    elif index_type == 'IP':
        index = faiss.IndexFlatIP(dim)  # Produit scalaire (cosinus si normalisés)
    elif index_type == 'IVF':
        # Index IVF avec quantization plus rapide mais moins précis
        nlist = min(4096, 4 * int(np.sqrt(embeddings.shape[0])))  # nombre de cellules Voronoi
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings)
        index.nprobe = 16  # nombre de cellules à explorer pour la recherche (compromis vitesse/précision)
    else:
        # Par défaut: index plat L2
        index = faiss.IndexFlatL2(dim)
    
    # Ajout des vecteurs à l'index
    index.add(embeddings.astype(np.float32))  # FAISS requiert des float32
    
    # Préparation des métadonnées
    if metadata is None:
        metadata = [{} for _ in chunks]
    
    index_metadata = {
        'chunks': chunks,
        'metadata': metadata,
        'dim': dim,
        'index_type': index_type,
        'count': len(chunks)
    }
    
    print(f"Index FAISS créé avec {len(chunks)} vecteurs de dimension {dim}")
    return index, index_metadata

def save_faiss_index(
    index: faiss.Index,
    index_metadata: Dict[str, Any],
    file_path: str
) -> bool:
    """
    Sauvegarde l'index FAISS et ses métadonnées sur disque.
    
    Args:
        index (faiss.Index): Index FAISS à sauvegarder
        index_metadata (Dict): Dictionnaire de métadonnées
        file_path (str): Chemin où sauvegarder l'index
        
    Returns:
        bool: True si sauvegarde réussie
    """
    try:
        # Assurer que le répertoire existe
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Sauvegarder l'index FAISS
        faiss.write_index(index, f"{file_path}.index")
        
        # Sauvegarder les métadonnées
        with open(f"{file_path}.meta", 'wb') as f:
            pickle.dump(index_metadata, f)
            
        print(f"Index FAISS et métadonnées sauvegardés dans '{file_path}.index' et '{file_path}.meta'")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'index: {str(e)}")
        return False

def load_faiss_index(file_path: str) -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Charge un index FAISS et ses métadonnées depuis le disque.
    
    Args:
        file_path (str): Chemin de base des fichiers d'index
        
    Returns:
        Tuple[faiss.Index, Dict]: Index FAISS et dictionnaire de métadonnées
    """
    try:
        # Charger l'index FAISS
        index = faiss.read_index(f"{file_path}.index")
        
        # Charger les métadonnées
        with open(f"{file_path}.meta", 'rb') as f:
            index_metadata = pickle.load(f)
            
        print(f"Index FAISS chargé avec {index.ntotal} vecteurs")
        return index, index_metadata
    except Exception as e:
        print(f"Erreur lors du chargement de l'index: {str(e)}")
        raise

def search_similar(
    query: str,
    index: faiss.Index,
    index_metadata: Dict[str, Any],
    model: Optional[SentenceTransformer] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Recherche les chunks les plus similaires à une requête.
    
    Args:
        query (str): Texte de la requête
        index (faiss.Index): Index FAISS
        index_metadata (Dict): Métadonnées de l'index
        model (SentenceTransformer, optional): Modèle préchargé ou None pour en charger un nouveau
        model_name (str): Nom du modèle à utiliser si model=None
        top_k (int): Nombre de résultats à retourner
        
    Returns:
        List[Dict]: Liste des résultats avec scores de similarité
    """
    # Charger le modèle si non fourni
    if model is None:
        model = load_embedding_model(model_name)
    
    # Convertir la requête en vecteur d'embedding
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    # Recherche des plus proches voisins
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    
    # Récupérer les résultats
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx != -1:  # FAISS peut retourner -1 si pas assez de résultats
            results.append({
                'chunk': index_metadata['chunks'][idx],
                'score': float(1.0 - distances[0][i] / 2) if index_metadata['index_type'] == 'L2' else float(distances[0][i]),
                'index': int(idx),
                'metadata': index_metadata['metadata'][idx] if idx < len(index_metadata['metadata']) else {}
            })
    
    return results

def process_and_index_chunks(
    chunks: List[str],
    output_path: Optional[str] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    index_type: str = 'L2',
    metadata: Optional[List[Dict[str, Any]]] = None
) -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Traite une liste de chunks, génère les embeddings et crée un index FAISS.
    
    Args:
        chunks (List[str]): Liste de chunks de texte
        output_path (str, optional): Chemin où sauvegarder l'index (si None, pas de sauvegarde)
        model_name (str): Nom du modèle sentence-transformers à utiliser
        index_type (str): Type d'index FAISS à créer
        metadata (List[Dict], optional): Métadonnées pour chaque chunk
        
    Returns:
        Tuple[faiss.Index, Dict]: Index FAISS et métadonnées
    """
    print(f"Traitement de {len(chunks)} chunks...")
    
    # Charger le modèle d'embedding
    model = load_embedding_model(model_name)
    
    # Générer les embeddings
    embeddings = chunks_to_embeddings(chunks, model)
    
    # Créer l'index FAISS
    index, index_metadata = create_faiss_index(embeddings, chunks, index_type, metadata)
    
    # Sauvegarder l'index si un chemin est fourni
    if output_path:
        save_faiss_index(index, index_metadata, output_path)
    
    return index, index_metadata
