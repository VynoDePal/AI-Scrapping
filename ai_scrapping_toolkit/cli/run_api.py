#!/usr/bin/env python3
"""
Script pour lancer l'API AI Scrapping avec uvicorn.
"""

import uvicorn
import argparse
import os
import logging

# Essayer de charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Variables d'environnement chargées depuis .env")
except ImportError:
    print("Le package python-dotenv n'est pas installé. Les variables d'environnement .env ne seront pas chargées.")
    print("Pour l'installer: pip install python-dotenv")

def main():
    parser = argparse.ArgumentParser(description="Lance l'API AI Scrapping")
    parser.add_argument('--host', default='127.0.0.1', help='Adresse d\'écoute (défaut: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='Port d\'écoute (défaut: 8000)')
    parser.add_argument('--reload', action='store_true', help='Activer le rechargement automatique du code (développement)')
    parser.add_argument('--workers', type=int, default=1, help='Nombre de workers (défaut: 1)')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], 
                       help='Niveau de journalisation (défaut: info)')
    
    args = parser.parse_args()
    
    print(f"Démarrage de l'API AI Scrapping sur {args.host}:{args.port}")
    print(f"Documentation Swagger accessible à http://{args.host}:{args.port}/docs")
    
    if args.reload:
        print("Mode rechargement automatique activé (développement)")
    
    uvicorn.run(
        "ai_scrapping_toolkit.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()
