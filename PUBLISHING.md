# Guide de publication sur PyPI

Ce document explique comment publier le package AI Scrapping Toolkit sur PyPI.

## Prérequis

1. Créer un compte sur PyPI: https://pypi.org/account/register/
2. Installer les outils nécessaires:
   ```bash
   pip install --upgrade pip build twine
   ```

## Étapes de publication

### 1. Préparation

Assurez-vous que votre version est correctement définie dans `setup.py`. Incrémentez le numéro de version à chaque nouvelle publication.

### 2. Construction du package

```bash
# Nettoyer les anciens builds
rm -rf build/ dist/ *.egg-info/

# Construire le package
python -m build
```

Cette commande génère des fichiers dans le dossier `dist/`:
- Un archive source (`.tar.gz`)
- Une distribution wheel (`.whl`)

### 3. Test du package (recommandé)

Avant de publier sur PyPI officiel, testez sur Test PyPI:

```bash
# Upload sur Test PyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Puis testez l'installation:

```bash
pip install --index-url https://test.pypi.org/simple/ ai-scrapping-toolkit
```

### 4. Publication sur PyPI officiel

Une fois testé, publiez sur PyPI officiel:

```bash
twine upload dist/*
```

Utilisez vos identifiants PyPI lorsque demandés, ou créez un fichier `.pypirc` dans votre répertoire personnel:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = votre_nom_utilisateur
password = votre_mot_de_passe

[testpypi]
repository = https://test.pypi.org/legacy/
username = votre_nom_utilisateur
password = votre_mot_de_passe
```

### 5. Vérification

Vérifiez que votre package est correctement installable:

```bash
pip install ai-scrapping-toolkit
```

## Publication automatisée avec le script

Vous pouvez utiliser le script `pypi_publish.sh` pour automatiser une partie du processus:

```bash
chmod +x pypi_publish.sh
./pypi_publish.sh
```

## Tokens d'API (recommandé pour plus de sécurité)

Au lieu d'utiliser votre mot de passe, vous pouvez créer des tokens d'API sur PyPI:

1. Connectez-vous à votre compte PyPI
2. Allez dans "Account settings" > "API tokens"
3. Créez un nouveau token avec les permissions appropriées
4. Utilisez ce token comme mot de passe lors de l'upload

## Références

- [Documentation PyPI sur le packaging](https://packaging.python.org/tutorials/packaging-projects/)
- [Documentation de Twine](https://twine.readthedocs.io/en/latest/)
