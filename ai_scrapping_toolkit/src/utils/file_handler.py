"""
Module pour la gestion des fichiers dans le projet AI Scrapping.
"""

import os
import json

def ensure_directory_exists(filepath):
    """
    Assure que le répertoire parent du fichier existe.
    
    Args:
        filepath (str): Chemin du fichier
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_file(content, filepath, encoding='utf-8'):
    """
    Sauvegarde du contenu dans un fichier.
    
    Args:
        content (str): Contenu à sauvegarder
        filepath (str): Chemin du fichier de destination
        encoding (str): Encodage à utiliser
    
    Returns:
        bool: True si la sauvegarde a réussi, False sinon
    """
    try:
        ensure_directory_exists(filepath)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier: {e}")
        return False

def load_file(filepath, encoding='utf-8'):
    """
    Charge le contenu d'un fichier.
    
    Args:
        filepath (str): Chemin du fichier à charger
        encoding (str): Encodage à utiliser
    
    Returns:
        str ou None: Contenu du fichier ou None en cas d'erreur
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return None

def save_chunks(chunks, full_content, filename):
    """
    Sauvegarde les chunks en format JSON et optionnellement le contenu complet.
    
    Args:
        chunks (list): Liste des chunks de texte
        full_content (str): Contenu complet
        filename (str): Nom de base du fichier
        
    Returns:
        tuple: (bool, str) - Succès de l'opération et nom du fichier de chunks
    """
    try:
        # Sauvegarder en format JSON pour préserver la structure
        base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
        chunks_filename = f"{base_filename}_chunks.json"
        
        ensure_directory_exists(chunks_filename)
        with open(chunks_filename, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"{len(chunks)} chunks sauvegardés dans {chunks_filename}")
        
        # Sauvegarder aussi le texte complet si demandé
        if filename != chunks_filename:
            save_file(full_content, filename)
            print(f"Contenu complet sauvegardé dans {filename}")
        
        return True, chunks_filename
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des chunks: {e}")
        return False, None
