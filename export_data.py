#!/usr/bin/env python3
"""
Script pour organiser des données extraites/traitées en DataFrame pandas et les exporter en CSV.
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(source: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Charge des données à partir d'un fichier JSON ou d'un dictionnaire/liste.
    
    Args:
        source: Chemin du fichier JSON ou dictionnaire/liste de données
        
    Returns:
        Dict[str, Any]: Données chargées
    """
    # Si source est déjà un dictionnaire ou une liste
    if isinstance(source, (dict, list)):
        return source
    
    # Si source est un chemin de fichier
    if isinstance(source, str):
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Le fichier '{source}' n'est pas un JSON valide")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"Fichier '{source}' non trouvé")
            sys.exit(1)
    
    logger.error("Format de source non pris en charge")
    sys.exit(1)

def create_dataframe(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Crée un DataFrame pandas à partir de données structurées.
    
    Args:
        data: Données structurées (dict avec listes ou liste de dicts)
        
    Returns:
        pd.DataFrame: DataFrame créé
    """
    # Cas 1: data est un dict avec des listes (format de sortie d'aggregate_extraction_results)
    # Ex: {"titres": ["titre1", "titre2"], "dates": ["date1", "date2"]}
    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values() if v):
        # Vérifier si toutes les listes ont la même longueur
        list_lengths = [len(v) for v in data.values() if isinstance(v, list)]
        if list_lengths and all(x == list_lengths[0] for x in list_lengths):
            return pd.DataFrame(data)
        else:
            # Si les listes ont des longueurs différentes, on les aligne
            max_length = max(list_lengths) if list_lengths else 0
            aligned_data = {}
            
            for key, value in data.items():
                if isinstance(value, list):
                    # Étendre la liste avec None si nécessaire
                    aligned_data[key] = value + [None] * (max_length - len(value))
                else:
                    # Inclure les valeurs non-liste comme des colonnes avec valeur constante
                    aligned_data[key] = [value] * max_length
            
            return pd.DataFrame(aligned_data)
    
    # Cas 2: data est une liste de dictionnaires
    # Ex: [{"titre": "titre1", "date": "date1"}, {"titre": "titre2", "date": "date2"}]
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return pd.DataFrame(data)
    
    # Cas 3: data est un dictionnaire de dictionnaires
    # Ex: {"item1": {"titre": "titre1", "date": "date1"}, "item2": {"titre": "titre2", "date": "date2"}}
    elif isinstance(data, dict) and all(isinstance(item, dict) for item in data.values()):
        # Convertir en liste de dictionnaires avec une colonne d'ID
        items = []
        for key, value in data.items():
            item = value.copy()
            item['id'] = key
            items.append(item)
        return pd.DataFrame(items)
    
    # Cas particulier: data est une liste non vide mais pas de dicts
    elif isinstance(data, list) and data:
        if all(isinstance(item, (str, int, float)) for item in data):
            # Liste de valeurs simples
            return pd.DataFrame({"valeur": data})
        else:
            # Essayer de convertir chaque élément en dict
            try:
                return pd.DataFrame([dict(item) if hasattr(item, '__iter__') else {"valeur": item} for item in data])
            except (ValueError, TypeError):
                logger.error("Impossible de convertir les données en DataFrame")
                return pd.DataFrame({"donnees_brutes": [str(data)]})
    
    # Cas par défaut: créer un DataFrame avec une seule ligne
    logger.warning("Format de données non standard, conversion en DataFrame simplifiée")
    return pd.DataFrame([data] if not isinstance(data, list) else data)

def clean_dataframe(df: pd.DataFrame, options: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Nettoie et formate un DataFrame.
    
    Args:
        df: DataFrame à nettoyer
        options: Options de nettoyage
        
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    if options is None:
        options = {}
    
    # Créer une copie pour éviter de modifier l'original
    df_clean = df.copy()
    
    # Renommer les colonnes si spécifié
    if 'rename_columns' in options and isinstance(options['rename_columns'], dict):
        df_clean = df_clean.rename(columns=options['rename_columns'])
    
    # Supprimer les lignes avec trop de valeurs manquantes
    if options.get('drop_na_threshold'):
        threshold = options['drop_na_threshold']
        df_clean = df_clean.dropna(thresh=int(len(df_clean.columns) * threshold))
    
    # Supprimer les doublons en tenant compte des objets non hashables comme les dictionnaires
    if options.get('remove_duplicates', True):
        # Vérifier si des objets non hashables sont présents
        non_hashable_columns = []
        for col in df_clean.columns:
            if df_clean[col].apply(lambda x: isinstance(x, (dict, list))).any():
                non_hashable_columns.append(col)
        
        if non_hashable_columns:
            # Créer des versions stringifiées des colonnes non hashables
            temp_df = df_clean.copy()
            for col in non_hashable_columns:
                temp_df[f"{col}_str"] = temp_df[col].apply(lambda x: json.dumps(x) if x is not None else None)
            
            # Supprimer les doublons en utilisant les colonnes stringifiées
            string_cols = [f"{col}_str" for col in non_hashable_columns]
            other_cols = [col for col in df_clean.columns if col not in non_hashable_columns]
            subset_cols = other_cols + string_cols
            
            # Trouver les indices à conserver après déduplication
            keep_indices = ~temp_df.duplicated(subset=subset_cols, keep='first')
            
            # Appliquer le filtre sur le DataFrame original
            df_clean = df_clean.loc[keep_indices].reset_index(drop=True)
            
            logging.info(f"Suppression des doublons effectuée en utilisant les conversions JSON pour {len(non_hashable_columns)} colonnes non hashables")
        else:
            # Si aucun objet non hashable n'est présent, utiliser drop_duplicates() standard
            df_clean = df_clean.drop_duplicates()
    
    # Convertir les dates si des colonnes de date sont spécifiées
    if 'date_columns' in options:
        for col in options['date_columns']:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    
                    # Formater la date si un format est spécifié
                    if 'date_format' in options:
                        df_clean[col] = df_clean[col].dt.strftime(options['date_format'])
                except Exception as e:
                    logger.warning(f"Erreur lors de la conversion de la colonne date '{col}': {str(e)}")
    
    # Trier par une colonne si spécifié
    if 'sort_by' in options:
        sort_col = options['sort_by']
        ascending = options.get('sort_ascending', True)
        
        if sort_col in df_clean.columns:
            df_clean = df_clean.sort_values(by=sort_col, ascending=ascending)
    
    # Filtrer les colonnes si spécifié
    if 'columns' in options and isinstance(options['columns'], list):
        cols_to_keep = [col for col in options['columns'] if col in df_clean.columns]
        if cols_to_keep:
            df_clean = df_clean[cols_to_keep]
    
    return df_clean

def flatten_complex_data(df: pd.DataFrame, max_depth: int = 2) -> pd.DataFrame:
    """
    Aplatit les structures de données complexes (dictionnaires, listes) dans un DataFrame.
    
    Args:
        df: DataFrame à aplatir
        max_depth: Profondeur maximale d'aplatissement
        
    Returns:
        pd.DataFrame: DataFrame avec structures aplaties
    """
    flat_df = df.copy()
    
    # Fonction récursive pour aplatir une valeur
    def flatten_value(value, prefix='', depth=0):
        if depth >= max_depth:
            return {prefix: str(value) if value is not None else None}
        
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                key = f"{prefix}_{k}" if prefix else k
                result.update(flatten_value(v, key, depth + 1))
            return result
        elif isinstance(value, list):
            if not value:
                return {prefix: "[]"}
            
            if all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
                return {prefix: json.dumps(value)}
            
            result = {}
            for i, v in enumerate(value[:5]):  # Limiter à 5 éléments pour éviter explosion
                key = f"{prefix}_{i}" if prefix else f"item_{i}"
                result.update(flatten_value(v, key, depth + 1))
            
            if len(value) > 5:
                result[f"{prefix}_more"] = f"et {len(value) - 5} autres éléments"
                
            return result
        else:
            return {prefix: value}
    
    # Détecter et aplatir les colonnes avec des structures complexes
    complex_columns = []
    for col in flat_df.columns:
        if flat_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            complex_columns.append(col)
    
    if complex_columns:
        logging.info(f"Aplatissement de {len(complex_columns)} colonnes avec structures de données complexes")
        flat_data = []
        
        for _, row in flat_df.iterrows():
            row_dict = {}
            for col in flat_df.columns:
                if col in complex_columns:
                    flattened = flatten_value(row[col], col)
                    row_dict.update(flattened)
                else:
                    row_dict[col] = row[col]
            flat_data.append(row_dict)
        
        # Créer un nouveau DataFrame à partir des données aplaties
        return pd.DataFrame(flat_data)
    
    # Si aucune colonne complexe, retourner le DataFrame original
    return flat_df

def export_dataframe(
    data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    output_file: str = 'donnees.csv',
    options: Dict[str, Any] = None
) -> str:
    """
    Exporte des données structurées en fichier CSV via pandas DataFrame.
    
    Args:
        data: Données à exporter (chemin de fichier JSON ou données Python)
        output_file: Chemin du fichier CSV à générer
        options: Options de nettoyage et d'exportation
        
    Returns:
        str: Chemin du fichier CSV généré
    """
    if options is None:
        options = {}
    
    # Charger les données
    loaded_data = load_data(data)
    
    # Créer un DataFrame
    df = create_dataframe(loaded_data)
    logger.info(f"DataFrame créé avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Nettoyer et formater le DataFrame
    df_clean = clean_dataframe(df, options)
    logger.info(f"DataFrame nettoyé: {len(df_clean)} lignes restantes")
    
    # Aplatir les structures complexes si l'option est activée
    if options.get('flatten_complex', True):
        df_clean = flatten_complex_data(df_clean)
        logger.info(f"Structures de données complexes aplaties pour faciliter l'exportation CSV")
    
    # Assurer que le répertoire de sortie existe
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Vérifier s'il y a des objets non exportables en CSV
    has_complex_objects = False
    for col in df_clean.columns:
        if df_clean[col].apply(lambda x: isinstance(x, (dict, list))).any():
            has_complex_objects = True
            logger.warning(f"La colonne '{col}' contient des objets complexes qui seront convertis en chaînes")
            
            # Convertir les objets complexes en chaînes JSON
            df_clean[col] = df_clean[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
            )
    
    # Exporter en CSV
    try:
        csv_options = options.get('csv_options', {})
        df_clean.to_csv(output_file, index=options.get('include_index', False), **csv_options)
        logger.info(f"Données exportées avec succès dans {output_file}")
        return os.path.abspath(output_file)
    except Exception as e:
        logger.error(f"Erreur lors de l'exportation en CSV: {str(e)}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Organise et exporte des données en format CSV")
    
    parser.add_argument("input_file", help="Fichier JSON contenant les données à exporter")
    parser.add_argument("--output", "-o", default="donnees.csv", help="Nom du fichier CSV à générer")
    
    # Options de nettoyage et formatage
    parser.add_argument("--no-duplicates", action="store_true", help="Supprimer les doublons")
    parser.add_argument("--date-columns", nargs="+", help="Colonnes à convertir en dates")
    parser.add_argument("--date-format", default="%Y-%m-%d", help="Format des dates à la sortie")
    parser.add_argument("--sort-by", help="Colonne pour trier les données")
    parser.add_argument("--desc", action="store_true", help="Trier par ordre décroissant")
    parser.add_argument("--columns", nargs="+", help="Colonnes à inclure dans le résultat")
    
    # Options d'export CSV
    parser.add_argument("--index", action="store_true", help="Inclure la colonne d'index")
    parser.add_argument("--delimiter", default=",", help="Délimiteur CSV")
    parser.add_argument("--encoding", default="utf-8", help="Encodage du fichier CSV")
    
    # Prévisualisation
    parser.add_argument("--preview", action="store_true", help="Afficher un aperçu des données")
    # Modification: transformer --head en flag et ajouter --num-rows pour spécifier le nombre de lignes
    parser.add_argument("--head", action="store_true", help="Afficher les premières lignes (équivalent à --preview)")
    parser.add_argument("--num-rows", type=int, default=5, help="Nombre de lignes à afficher pour l'aperçu (défaut: 5)")
    
    # Options de formatage avancées
    parser.add_argument("--flatten", action="store_true", help="Aplatir les structures de données complexes")
    parser.add_argument("--max-flatten-depth", type=int, default=2, help="Profondeur maximale d'aplatissement")
    
    args = parser.parse_args()
    
    # Préparer les options
    options = {
        'remove_duplicates': args.no_duplicates,
        'include_index': args.index,
        'flatten_complex': args.flatten,
        'max_flatten_depth': args.max_flatten_depth,
        'csv_options': {
            'sep': args.delimiter,
            'encoding': args.encoding
        }
    }
    
    if args.date_columns:
        options['date_columns'] = args.date_columns
        options['date_format'] = args.date_format
    
    if args.sort_by:
        options['sort_by'] = args.sort_by
        options['sort_ascending'] = not args.desc
    
    if args.columns:
        options['columns'] = args.columns
    
    # Exporter les données
    output_path = export_dataframe(args.input_file, args.output, options)
    
    # Si --head est utilisé, activer aussi preview
    preview_enabled = args.preview or args.head
    
    if preview_enabled and output_path:
        try:
            df = pd.read_csv(output_path)
            print("\nAperçu des données exportées:")
            print(df.head(args.num_rows))
            print(f"\nTotal: {len(df)} lignes, {len(df.columns)} colonnes")
        except Exception as e:
            logger.error(f"Impossible d'afficher l'aperçu: {str(e)}")

if __name__ == "__main__":
    main()
