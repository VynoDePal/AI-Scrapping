"""
Utilitaires pour le projet AI Scrapping.
"""

from .file_handler import save_chunks, save_file, load_file, ensure_directory_exists

# Ajout de l'utilitaire d'export CSV pour l'utilisation programmatique
try:
    import pandas as pd
    from ..processors.data_processor import convert_to_dataframe
    
    def export_to_csv(data, output_file='donnees.csv', **options):
        """
        Exporte des données en CSV via pandas DataFrame.
        
        Args:
            data: Données à exporter (dict, list, DataFrame)
            output_file: Chemin du fichier CSV
            **options: Options pour pandas.to_csv()
            
        Returns:
            str: Chemin du fichier CSV généré
        """
        # Si c'est déjà un DataFrame, l'utiliser directement
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            # Sinon, convertir en DataFrame
            df = convert_to_dataframe(data)
            
        # Exporter en CSV
        df.to_csv(output_file, **options)
        return output_file
        
except ImportError:
    # pandas n'est pas disponible
    def export_to_csv(*args, **kwargs):
        raise ImportError("pandas est requis pour utiliser export_to_csv")

__all__ = [
    'save_chunks', 'save_file', 'load_file', 
    'ensure_directory_exists', 'export_to_csv'
]
