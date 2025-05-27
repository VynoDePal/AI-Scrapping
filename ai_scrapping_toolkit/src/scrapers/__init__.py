"""
Package de fonctions pour le scraping web.
"""

from .scraper import fetch_content
from .robots_checker import RobotsChecker

__all__ = ['fetch_content', 'RobotsChecker']
