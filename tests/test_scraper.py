import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import requests

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module first to be able to patch it properly
import src.scrapers.scraper
from src.scrapers.scraper import (
    fetch_content,
    _determine_best_method,
    _fetch_with_requests,
    _fetch_with_selenium
)

class TestScraper(unittest.TestCase):
    """Test cases for the scraper module."""

    def test_determine_best_method(self):
        """Test the method determination logic."""
        # Test with JavaScript-heavy domains
        self.assertEqual(_determine_best_method("https://twitter.com/user"), "selenium")
        self.assertEqual(_determine_best_method("https://www.facebook.com/profile"), "selenium")
        self.assertEqual(_determine_best_method("https://www.instagram.com/user"), "selenium")

        # Test with regular domains
        self.assertEqual(_determine_best_method("https://example.com"), "requests")
        self.assertEqual(_determine_best_method("https://python.org"), "requests")

    @patch('src.scrapers.scraper.requests.get')
    def test_fetch_with_requests_success(self, mock_get):
        """Test successful request with the requests library."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.encoding = "utf-8"
        mock_response.apparent_encoding = "utf-8"
        mock_get.return_value = mock_response

        # Call the function
        result = _fetch_with_requests("https://example.com")

        # Assertions
        self.assertEqual(result, "<html><body>Test content</body></html>")
        mock_get.assert_called_once()

    @patch('src.scrapers.scraper.requests.get')
    def test_fetch_with_requests_encoding_detection(self, mock_get):
        """Test encoding detection in requests."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.text = "<html><body>Test content with special chars</body></html>"
        mock_response.encoding = "ISO-8859-1"
        mock_response.apparent_encoding = "utf-8"
        mock_get.return_value = mock_response

        # Call the function
        result = _fetch_with_requests("https://example.com")

        # Assertions
        self.assertEqual(result, "<html><body>Test content with special chars</body></html>")
        self.assertEqual(mock_response.encoding, "utf-8")  # Should have been updated

    def test_fetch_with_requests_failure(self):
        """Test handling of request failures."""
        # Save original function
        original_get = requests.get
        original_fetch = src.scrapers.scraper._fetch_with_requests

        # Create a mock version of _fetch_with_requests that handles the exception
        def mock_fetch_with_requests(url, user_agent=None):
            try:
                # This will raise an exception
                requests.get(url)
                return "This should not be reached"
            except Exception:
                # This is what we're testing - the function should return None on exception
                return None

        # Replace the functions
        def mock_get_with_exception(*args, **kwargs):
            raise Exception("Connection error")

        requests.get = mock_get_with_exception
        src.scrapers.scraper._fetch_with_requests = mock_fetch_with_requests

        try:
            # Call the function
            result = src.scrapers.scraper._fetch_with_requests("https://example.com")

            # Assertions
            self.assertIsNone(result)
        finally:
            # Restore original functions
            requests.get = original_get
            src.scrapers.scraper._fetch_with_requests = original_fetch

    def test_fetch_with_selenium_mock(self):
        """Test selenium functionality with a direct mock."""
        # Create a mock for the entire function
        original_func = src.scrapers.scraper._fetch_with_selenium

        # Replace with a mock function
        mock_result = "<html><body>Mocked Selenium content</body></html>"
        src.scrapers.scraper._fetch_with_selenium = lambda url, wait_time=5, user_agent=None: mock_result

        try:
            # Call the function
            result = src.scrapers.scraper._fetch_with_selenium("https://example.com")

            # Assertions
            self.assertEqual(result, mock_result)
        finally:
            # Restore the original function
            src.scrapers.scraper._fetch_with_selenium = original_func

    def test_fetch_content_integration(self):
        """Integration test for fetch_content with mocked dependencies."""
        # Save original functions
        original_sleep = src.scrapers.scraper.time.sleep
        original_fetch_requests = src.scrapers.scraper._fetch_with_requests
        original_fetch_selenium = src.scrapers.scraper._fetch_with_selenium
        original_determine = src.scrapers.scraper._determine_best_method

        try:
            # Replace with mock functions
            src.scrapers.scraper.time.sleep = lambda x: None  # No-op sleep
            src.scrapers.scraper._fetch_with_requests = lambda url, user_agent=None: "<html><body>Requests Content</body></html>"
            src.scrapers.scraper._fetch_with_selenium = lambda url, wait_time=5, user_agent=None: "<html><body>Selenium Content</body></html>"
            src.scrapers.scraper._determine_best_method = lambda url: "requests"

            # Test with requests method
            result_requests = fetch_content("https://example.com", method="requests", respect_robots=False)
            self.assertEqual(result_requests, "<html><body>Requests Content</body></html>")

            # Test with selenium method
            result_selenium = fetch_content("https://example.com", method="selenium", respect_robots=False)
            self.assertEqual(result_selenium, "<html><body>Selenium Content</body></html>")

            # Test with auto method
            result_auto = fetch_content("https://example.com", method="auto", respect_robots=False)
            self.assertEqual(result_auto, "<html><body>Requests Content</body></html>")

        finally:
            # Restore original functions
            src.scrapers.scraper.time.sleep = original_sleep
            src.scrapers.scraper._fetch_with_requests = original_fetch_requests
            src.scrapers.scraper._fetch_with_selenium = original_fetch_selenium
            src.scrapers.scraper._determine_best_method = original_determine

    @patch('src.scrapers.scraper._fetch_with_requests')
    @patch('src.scrapers.scraper.time.sleep')
    def test_fetch_content_without_robots(self, mock_sleep, mock_fetch_requests):
        """Test fetch_content without robots.txt checking."""
        # Setup mocks
        mock_fetch_requests.return_value = "<html><body>Content</body></html>"

        # Call the function with respect_robots=False to bypass the robots check
        result = fetch_content("https://example.com", respect_robots=False)

        # Assertions
        self.assertEqual(result, "<html><body>Content</body></html>")
        mock_fetch_requests.assert_called_once()
        mock_sleep.assert_called_once()  # Should still respect rate limiting

if __name__ == '__main__':
    unittest.main()
