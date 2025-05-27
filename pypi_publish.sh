#!/bin/bash
# Script pour préparer et publier le package sur PyPI

echo "Préparation de la publication du package AI Scrapping Toolkit sur PyPI"

# Installation des outils nécessaires
echo "Installation des outils de packaging..."
pip install --upgrade pip
pip install --upgrade build twine

# Nettoyage des anciens builds
echo "Nettoyage des anciens builds..."
rm -rf build/ dist/ *.egg-info/

# Construction du package (source et wheel)
echo "Construction du package..."
python -m build

# Vérification du package avec twine
echo "Vérification du package..."
twine check dist/*

echo ""
echo "=============================================="
echo "Package prêt pour la publication!"
echo "=============================================="
echo ""
echo "Pour publier sur PyPI Test (recommandé pour tester):"
echo "twine upload --repository-url https://test.pypi.org/legacy/ dist/*"
echo ""
echo "Pour publier sur PyPI officiel:"
echo "twine upload dist/*"
echo ""
echo "Vous devrez fournir vos identifiants PyPI lors de l'upload."
echo "Si vous n'avez pas encore de compte PyPI, créez-en un sur:"
echo "https://pypi.org/account/register/"
echo ""
