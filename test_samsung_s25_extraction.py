#!/usr/bin/env python3
"""
Script de test pour extraire des informations sur le Samsung Galaxy S25 Ultra
à partir de différentes sources web en utilisant l'AI Scrapping Toolkit.
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Charger les variables d'environnement depuis .env
load_dotenv()
api_keys = {
    key: os.environ.get(key) for key in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "HUGGINGFACE_API_KEY"]
}
logger.info(f"Variables d'environnement chargées: {', '.join([k for k, v in api_keys.items() if v])}")

# Importer les modules nécessaires
from src.scrapers import fetch_content
from src.processors import extract_main_content, html_to_chunks
from src.llm import get_llm_provider, extract_data_from_chunks, aggregate_extraction_results

# URLs à scraper
URLS = {
    "frandroid": "https://www.frandroid.com/marques/samsung/2482652_test-samsung-galaxy-s25-ultra",
    "cdiscount": "https://www.cdiscount.com/telephonie/r-samsung+galaxy+s25+ultra.html#_his_"
}

# Configuration du LLM
LLM_CONFIG = {
    "provider": "openrouter",  # Utiliser OpenRouter comme provider
    "model": "meta-llama/llama-4-scout",  # Modèle spécifié dans les instructions
    "api_key": "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",  # Clé API fournie
    "temperature": 0.0,
    "max_tokens": 2048,
    "timeout": 180
}

# Requêtes d'extraction pour chaque site
EXTRACTION_QUERIES = {
    "frandroid": """
    Extrait les informations suivantes de cet article de test sur le Samsung Galaxy S25 Ultra:
    1. Caractéristiques techniques complètes (écran, processeur, RAM, stockage, batterie, etc.)
    2. Points forts du téléphone mentionnés dans l'article
    3. Points faibles ou critiques mentionnés dans l'article
    4. Note globale donnée par le testeur (si disponible)
    5. Prix et disponibilité mentionnés
    Organise ces informations dans un JSON structuré avec les clés suivantes:
    - caracteristiques_techniques (objet avec sous-catégories)
    - points_forts (liste)
    - points_faibles (liste)
    - note_globale (nombre ou texte)
    - prix_et_disponibilite (objet)
    """,

    "cdiscount": """
    Extrait les informations suivantes de cette page de résultats de recherche pour le Samsung Galaxy S25 Ultra:
    1. Liste des produits affichés (nom, prix, description courte)
    2. Fourchette de prix (prix minimum et maximum)
    3. Options de filtrage disponibles
    4. Promotions ou offres spéciales mentionnées
    Organise ces informations dans un JSON structuré avec les clés suivantes:
    - produits (liste d'objets avec nom, prix, description)
    - fourchette_prix (objet avec min et max)
    - filtres_disponibles (liste)
    - promotions (liste)
    """
}

def ensure_output_dir():
    """Crée le répertoire de sortie s'il n'existe pas."""
    output_dir = "resultats_samsung_s25"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def process_website(site_name, url, query):
    """Traite un site web: scraping, extraction et analyse."""
    logger.info(f"\n==================================================")
    logger.info(f"Traitement de {site_name}: {url}")
    logger.info(f"==================================================")

    output_dir = ensure_output_dir()

    # Configuration spécifique pour chaque site
    scraping_config = {}
    if site_name == "frandroid":
        logger.info("Ignorer robots.txt pour frandroid.com (uniquement pour ce test)")
        scraping_config = {
            "respect_robots": False,
            "method": "requests"
        }
    elif site_name == "cdiscount":
        logger.info("Utilisation de Selenium pour cdiscount.com")
        scraping_config = {
            "method": "selenium",
            "wait_time": 10
        }

    # Récupération du contenu HTML
    logger.info(f"Récupération du contenu depuis {url}...")
    html_content = fetch_content(url, **scraping_config)

    if not html_content:
        logger.error(f"Échec de la récupération du contenu pour {site_name}")
        return None

    logger.info(f"Contenu récupéré avec succès ({len(html_content)} caractères)")

    # Sauvegarde du HTML brut
    raw_html_path = os.path.join(output_dir, f"{site_name}_raw.html")
    with open(raw_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"HTML brut sauvegardé dans {raw_html_path}")

    # Extraction du contenu principal
    logger.info("Extraction du contenu principal...")
    main_content = extract_main_content(html_content)

    # Sauvegarde du contenu principal
    main_content_path = os.path.join(output_dir, f"{site_name}_main.txt")
    with open(main_content_path, "w", encoding="utf-8") as f:
        f.write(main_content)
    logger.info(f"Contenu principal sauvegardé dans {main_content_path}")

    # Vérifier si le contenu principal est valide
    if site_name == "cdiscount" and "Accès non autorisé" in main_content:
        logger.error(f"Accès non autorisé à {site_name}. Le site a détecté notre scraping.")
        error_data = {"error": "access_denied", "message": "Le site a bloqué notre accès."}
        results_path = os.path.join(output_dir, f"{site_name}_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        return error_data

    # Découpage en chunks
    logger.info("Découpage du contenu en chunks...")
    chunks = html_to_chunks(main_content, method="hybrid", max_length=4000)
    logger.info(f"{len(chunks)} chunks générés")

    # Vérifier la clé API OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("Clé API OpenAI non trouvée dans les variables d'environnement")
        error_data = {"error": "api_key_missing", "message": "Clé API OpenAI manquante"}
        results_path = os.path.join(output_dir, f"{site_name}_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        return error_data
    else:
        # Afficher une version masquée de la clé API pour le débogage
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        logger.info(f"Clé API OpenAI trouvée (masquée): {masked_key}")

    # Initialisation du provider LLM
    logger.info(f"Initialisation du provider LLM ({LLM_CONFIG['provider']}/{LLM_CONFIG['model']})...")
    try:
        llm_provider = get_llm_provider(**LLM_CONFIG)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du provider LLM: {str(e)}")
        error_data = {"error": "llm_provider_init_failed", "message": str(e)}
        results_path = os.path.join(output_dir, f"{site_name}_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        return error_data

    # Extraction des données avec le LLM
    logger.info(f"Extraction des données avec la requête:\n    {query[:50]}...")
    logger.info(f"Prompt d'extraction:\n    {query}")

    # Utiliser un seul chunk pour tester
    test_chunk = chunks[0]
    logger.info(f"Test d'extraction sur le premier chunk ({len(test_chunk)} caractères)...")
    try:
        # Vérifier si nous utilisons OpenRouter
        if LLM_CONFIG["provider"] == "openrouter":
            # Utiliser directement l'API OpenRouter via requests
            import requests

            # Construire le système de messages
            system_prompt = (
                "Tu es un assistant spécialisé dans l'extraction de données à partir de contenu HTML. "
                "Analyse le contenu et extrait les informations demandées selon l'instruction. "
                "Réponds uniquement avec un objet JSON valide, sans texte avant ou après. "
                "N'utilise pas de bloc de code markdown. Commence directement par { et termine par }. "
                "Si tu ne trouves pas d'information, renvoie un objet avec des tableaux vides."
            )

            user_prompt = f"### Instruction:\n{query}\n\n### Contenu HTML à analyser:\n{test_chunk}"

            # Préparer les données de la requête
            payload = {
                "model": LLM_CONFIG["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": LLM_CONFIG["temperature"],
                "max_tokens": LLM_CONFIG["max_tokens"]
            }

            headers = {
                "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-scrapping-toolkit.com",
                "X-Title": "AI Scrapping Toolkit"
            }

            logger.info(f"Envoi de la requête à OpenRouter (modèle: {LLM_CONFIG['model']})...")
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=LLM_CONFIG["timeout"]
            )

            if response.status_code != 200:
                error_msg = f"Erreur OpenRouter: {response.status_code} - {response.text}"
                logger.error(error_msg)
                error_data = {"error": f"openrouter_error_{response.status_code}", "message": response.text}
                results_path = os.path.join(output_dir, f"{site_name}_results.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(error_data, f, ensure_ascii=False, indent=2)
                return error_data

            # Extraire la réponse
            response_data = response.json()
            result = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # Essayer de parser le JSON
            try:
                # Trouver le premier { et le dernier } au cas où il y aurait du texte avant/après
                first_brace = result.find("{")
                last_brace = result.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    result = result[first_brace:last_brace+1].strip()

                test_result = json.loads(result)
                logger.info(f"Résultat du test d'extraction: {json.dumps(test_result, ensure_ascii=False)[:200]}...")
            except json.JSONDecodeError as e:
                error_msg = f"Erreur de décodage JSON: {str(e)}. Réponse brute: {result[:200]}..."
                logger.error(error_msg)
                error_data = {"error": "json_decode_error", "message": str(e), "raw_response": result}
                results_path = os.path.join(output_dir, f"{site_name}_results.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(error_data, f, ensure_ascii=False, indent=2)
                return error_data
        else:
            # Utiliser le provider LLM standard
            test_result = llm_provider.extract(test_chunk, query)
            logger.info(f"Résultat du test d'extraction: {json.dumps(test_result, ensure_ascii=False)[:200]}...")

        if "error" in test_result:
            logger.error(f"Erreur lors du test d'extraction: {test_result['error']}")
            results_path = os.path.join(output_dir, f"{site_name}_results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(test_result, f, ensure_ascii=False, indent=2)
            return test_result
    except Exception as e:
        logger.error(f"Exception lors du test d'extraction: {str(e)}")
        import traceback
        logger.error(f"Détails de l'erreur: {traceback.format_exc()}")
        error_data = {"error": "extraction_test_failed", "message": str(e)}
        results_path = os.path.join(output_dir, f"{site_name}_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        return error_data

    # Continuer avec l'extraction complète
    try:
        # Si nous avons déjà un résultat de test réussi, utilisons-le comme premier résultat
        if 'test_result' in locals() and isinstance(test_result, dict) and "error" not in test_result:
            logger.info("Utilisation du résultat de test comme premier résultat d'extraction")
            # Créer un résultat agrégé à partir du test
            aggregated_data = test_result

            # Sauvegarde des résultats
            results_path = os.path.join(output_dir, f"{site_name}_results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Résultats sauvegardés dans {results_path}")

            return aggregated_data
        else:
            # Sinon, procéder à l'extraction complète
            logger.info("Procédant à l'extraction complète...")

            # Si nous utilisons OpenRouter, nous devons traiter chaque chunk manuellement
            if LLM_CONFIG["provider"] == "openrouter":
                logger.info(f"Extraction avec OpenRouter sur {len(chunks)} chunks...")

                # Traiter les chunks un par un pour éviter les erreurs
                all_results = []
                for i, chunk in enumerate(chunks[:min(5, len(chunks))]):  # Limiter à 5 chunks pour ce test
                    logger.info(f"Traitement du chunk {i+1}/{min(5, len(chunks))}...")

                    # Utiliser le même code que pour le test
                    import requests

                    # Construire le système de messages
                    system_prompt = (
                        "Tu es un assistant spécialisé dans l'extraction de données à partir de contenu HTML. "
                        "Analyse le contenu et extrait les informations demandées selon l'instruction. "
                        "Réponds uniquement avec un objet JSON valide, sans texte avant ou après. "
                        "N'utilise pas de bloc de code markdown. Commence directement par { et termine par }. "
                        "Si tu ne trouves pas d'information, renvoie un objet avec des tableaux vides."
                    )

                    user_prompt = f"### Instruction:\n{query}\n\n### Contenu HTML à analyser:\n{chunk}"

                    # Préparer les données de la requête
                    payload = {
                        "model": LLM_CONFIG["model"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": LLM_CONFIG["temperature"],
                        "max_tokens": LLM_CONFIG["max_tokens"]
                    }

                    headers = {
                        "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://ai-scrapping-toolkit.com",
                        "X-Title": "AI Scrapping Toolkit"
                    }

                    try:
                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=LLM_CONFIG["timeout"]
                        )

                        if response.status_code != 200:
                            logger.error(f"Erreur OpenRouter sur le chunk {i+1}: {response.status_code} - {response.text}")
                            continue

                        # Extraire la réponse
                        response_data = response.json()
                        result = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                        # Essayer de parser le JSON
                        try:
                            # Trouver le premier { et le dernier } au cas où il y aurait du texte avant/après
                            first_brace = result.find("{")
                            last_brace = result.rfind("}")
                            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                                result = result[first_brace:last_brace+1].strip()

                            chunk_result = json.loads(result)
                            all_results.append(chunk_result)
                            logger.info(f"Résultat du chunk {i+1} obtenu avec succès")
                        except json.JSONDecodeError as e:
                            logger.error(f"Erreur de décodage JSON sur le chunk {i+1}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement du chunk {i+1}: {str(e)}")

                # Agréger les résultats manuellement
                logger.info(f"Agrégation de {len(all_results)} résultats...")
                if not all_results:
                    error_data = {"error": "no_results", "message": "Aucun résultat d'extraction obtenu"}
                    results_path = os.path.join(output_dir, f"{site_name}_results.json")
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(error_data, f, ensure_ascii=False, indent=2)
                    return error_data

                # Utiliser le premier résultat comme base
                aggregated_data = all_results[0]

                # Sauvegarde des résultats
                results_path = os.path.join(output_dir, f"{site_name}_results.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Résultats sauvegardés dans {results_path}")

                return aggregated_data
            else:
                # Utiliser l'extraction standard
                extraction_results = extract_data_from_chunks(
                    chunks=chunks,
                    query=query,
                    llm_provider=llm_provider,
                    max_workers=min(4, len(chunks))
                )

                # Agrégation des résultats
                logger.info("Agrégation des résultats...")
                aggregated_data = aggregate_extraction_results(extraction_results)

                # Sauvegarde des résultats
                results_path = os.path.join(output_dir, f"{site_name}_results.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Résultats sauvegardés dans {results_path}")

                return aggregated_data
    except Exception as e:
        logger.error(f"Exception lors de l'extraction complète: {str(e)}")
        import traceback
        logger.error(f"Détails de l'erreur: {traceback.format_exc()}")
        error_data = {"error": "extraction_failed", "message": str(e)}
        results_path = os.path.join(output_dir, f"{site_name}_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        return error_data

def main():
    """Fonction principale."""
    logger.info("Démarrage du test d'extraction d'informations sur le Samsung Galaxy S25 Ultra")

    results = {}

    # Traiter chaque site
    for site_name, url in URLS.items():
        results[site_name] = process_website(site_name, url, EXTRACTION_QUERIES[site_name])

    # Afficher un résumé des résultats
    logger.info("\nRÉSUMÉ DES RÉSULTATS:")
    for site_name, data in results.items():
        if data and len(data) > 0:
            logger.info(f"\n{site_name.upper()}: Extraction réussie")
            for key, value in data.items():
                if isinstance(value, list):
                    logger.info(f"- {key}: {len(value)} éléments")
                else:
                    logger.info(f"- {key}: {type(value).__name__}")
        else:
            logger.info(f"\n{site_name.upper()}: Échec de l'extraction")

    logger.info("\nTest terminé. Consultez le dossier 'resultats_samsung_s25' pour les détails.")

if __name__ == "__main__":
    main()
