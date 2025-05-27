import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from fastapi.testclient import TestClient

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app
from api.models.requests import ScrapingRequest
from api.models.responses import ScrapingResponse

class TestScrapingRouter(unittest.TestCase):
    """Test cases for the scraping router."""

    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)

    @patch('api.routers.scraping.fetch_content')
    @patch('api.routers.scraping.get_page_title')
    def test_scrape_url_basic(self, mock_get_page_title, mock_fetch_content):
        """Test basic URL scraping functionality."""
        # Setup mocks
        mock_fetch_content.return_value = "<html><body>Test content</body></html>"
        mock_get_page_title.return_value = "Test Page"

        # Make request
        response = self.client.post(
            "/scraping/",
            json={
                "url": "https://example.com",
                "method": "requests",
                "wait_time": 5,
                "preprocess": False,
                "extract_main_content": False,
                "respect_robots": True,
                "rate_limit": 1.0
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # URL may have a trailing slash added by Pydantic HttpUrl validation
        self.assertTrue(data["url"] in ["https://example.com", "https://example.com/"])
        self.assertEqual(data["title"], "Test Page")
        self.assertEqual(data["content"], "<html><body>Test content</body></html>")
        self.assertEqual(data["content_type"], "HTML")
        self.assertTrue("timestamp" in data)
        self.assertTrue("content_length" in data)

        # Verify mocks were called correctly
        # URL may have a trailing slash added by Pydantic HttpUrl validation
        args, kwargs = mock_fetch_content.call_args
        self.assertTrue(args[0] in ["https://example.com", "https://example.com/"])
        self.assertEqual(kwargs["method"], "requests")
        self.assertEqual(kwargs["wait_time"], 5)
        self.assertEqual(kwargs["respect_robots"], True)
        self.assertEqual(kwargs["user_agent"], None)
        self.assertEqual(kwargs["rate_limit"], 1.0)
        mock_get_page_title.assert_called_once()

    @patch('api.routers.scraping.fetch_content')
    @patch('api.routers.scraping.preprocess_html')
    @patch('api.routers.scraping.get_page_title')
    def test_scrape_url_with_preprocessing(self, mock_get_page_title, mock_preprocess_html, mock_fetch_content):
        """Test URL scraping with preprocessing."""
        # Setup mocks
        mock_fetch_content.return_value = "<html><body>Test content</body></html>"
        mock_preprocess_html.return_value = "Preprocessed content"
        mock_get_page_title.return_value = "Test Page"

        # Make request
        response = self.client.post(
            "/scraping/",
            json={
                "url": "https://example.com",
                "preprocess": True,
                "extract_main_content": False
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["content"], "Preprocessed content")
        self.assertEqual(data["content_type"], "preprocessed")

        # Verify mocks were called correctly
        mock_fetch_content.assert_called_once()
        mock_preprocess_html.assert_called_once_with("<html><body>Test content</body></html>")
        mock_get_page_title.assert_called_once()

    @patch('api.routers.scraping.fetch_content')
    @patch('api.routers.scraping.extract_main_content')
    @patch('api.routers.scraping.get_page_title')
    def test_scrape_url_with_main_content_extraction(self, mock_get_page_title, mock_extract_main_content, mock_fetch_content):
        """Test URL scraping with main content extraction."""
        # Setup mocks
        mock_fetch_content.return_value = "<html><body>Test content</body></html>"
        mock_extract_main_content.return_value = "Main content"
        mock_get_page_title.return_value = "Test Page"

        # Make request
        response = self.client.post(
            "/scraping/",
            json={
                "url": "https://example.com",
                "preprocess": False,
                "extract_main_content": True
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["content"], "Main content")
        self.assertEqual(data["content_type"], "main_content")

        # Verify mocks were called correctly
        mock_fetch_content.assert_called_once()
        mock_extract_main_content.assert_called_once_with("<html><body>Test content</body></html>")
        mock_get_page_title.assert_called_once()

    @patch('api.routers.scraping.fetch_content')
    def test_scrape_url_fetch_failure(self, mock_fetch_content):
        """Test URL scraping when fetch_content fails."""
        # Setup mock to return None (fetch failure)
        mock_fetch_content.return_value = None

        # Make request
        response = self.client.post(
            "/scraping/",
            json={
                "url": "https://example.com"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertTrue("Impossible de récupérer le contenu" in data["detail"])

    @patch('api.routers.scraping.fetch_content')
    def test_scrape_url_exception(self, mock_fetch_content):
        """Test URL scraping when an exception occurs."""
        # Setup mock to raise an exception
        mock_fetch_content.side_effect = Exception("Test error")

        # Make request
        response = self.client.post(
            "/scraping/",
            json={
                "url": "https://example.com"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertTrue("Test error" in data["detail"])

    @patch('src.scrapers.robots_checker.RobotsChecker')
    def test_check_url_access_allowed(self, mock_robots_checker_class):
        """Test URL access check when access is allowed."""
        # Setup mock
        mock_robots_checker = MagicMock()
        mock_robots_checker.can_fetch.return_value = (True, None)
        mock_robots_checker_class.return_value = mock_robots_checker

        # Make request
        response = self.client.post(
            "/scraping/check",
            params={"url": "https://example.com", "check_robots": True}
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["url"], "https://example.com")
        self.assertTrue(data["accessible"])
        self.assertIsNone(data["reason"])
        self.assertTrue(data["checked_robots"])
        self.assertTrue("timestamp" in data)

        # Verify mock was called correctly
        mock_robots_checker_class.assert_called_once_with(respect_robots=True)
        mock_robots_checker.can_fetch.assert_called_once_with("https://example.com")

    @patch('src.scrapers.robots_checker.RobotsChecker')
    def test_check_url_access_disallowed(self, mock_robots_checker_class):
        """Test URL access check when access is disallowed."""
        # Setup mock
        mock_robots_checker = MagicMock()
        mock_robots_checker.can_fetch.return_value = (False, "Interdit par robots.txt")
        mock_robots_checker_class.return_value = mock_robots_checker

        # Make request
        response = self.client.post(
            "/scraping/check",
            params={"url": "https://example.com", "check_robots": True}
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["url"], "https://example.com")
        self.assertFalse(data["accessible"])
        self.assertEqual(data["reason"], "Interdit par robots.txt")
        self.assertTrue(data["checked_robots"])

    @patch('src.scrapers.robots_checker.RobotsChecker')
    def test_check_url_access_exception(self, mock_robots_checker_class):
        """Test URL access check when an exception occurs."""
        # Setup mock to raise an exception
        mock_robots_checker = MagicMock()
        mock_robots_checker.can_fetch.side_effect = Exception("Test error")
        mock_robots_checker_class.return_value = mock_robots_checker

        # Make request
        response = self.client.post(
            "/scraping/check",
            params={"url": "https://example.com", "check_robots": True}
        )

        # Assertions
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertTrue("Test error" in data["detail"])

if __name__ == '__main__':
    unittest.main()
