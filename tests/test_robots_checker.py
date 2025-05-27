import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import time

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scrapers.robots_checker import RobotsChecker

class TestRobotsChecker(unittest.TestCase):
    """Test cases for the robots_checker module."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        checker = RobotsChecker()
        self.assertEqual(checker.user_agent, "*")
        self.assertTrue(checker.respect_robots)
        self.assertEqual(checker.rate_limit, 1.0)
        self.assertEqual(checker.parsers, {})
        self.assertEqual(checker.last_request_time, {})

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        checker = RobotsChecker(
            user_agent="TestBot/1.0",
            respect_robots=False,
            rate_limit=2.5
        )
        self.assertEqual(checker.user_agent, "TestBot/1.0")
        self.assertFalse(checker.respect_robots)
        self.assertEqual(checker.rate_limit, 2.5)

    def test_can_fetch_respect_robots_false(self):
        """Test can_fetch when respect_robots is False."""
        checker = RobotsChecker(respect_robots=False)
        result, reason = checker.can_fetch("https://example.com/page")
        self.assertTrue(result)
        self.assertIsNone(reason)

    @patch('urllib.robotparser.RobotFileParser')
    def test_can_fetch_allowed(self, mock_parser_class):
        """Test can_fetch when access is allowed by robots.txt."""
        # Setup mock
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = True
        mock_parser.crawl_delay.return_value = None
        mock_parser_class.return_value = mock_parser

        # Create checker and test
        checker = RobotsChecker(user_agent="TestBot/1.0")
        result, reason = checker.can_fetch("https://example.com/page")

        # Assertions
        self.assertTrue(result)
        self.assertIsNone(reason)
        mock_parser.set_url.assert_called_once_with("https://example.com/robots.txt")
        mock_parser.read.assert_called_once()
        mock_parser.can_fetch.assert_called_once_with("TestBot/1.0", "https://example.com/page")

    @patch('urllib.robotparser.RobotFileParser')
    def test_can_fetch_disallowed(self, mock_parser_class):
        """Test can_fetch when access is disallowed by robots.txt."""
        # Setup mock
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = False
        mock_parser_class.return_value = mock_parser

        # Create checker and test
        checker = RobotsChecker()
        result, reason = checker.can_fetch("https://example.com/page")

        # Assertions
        self.assertFalse(result)
        self.assertEqual(reason, "Interdit par robots.txt")
        mock_parser.can_fetch.assert_called_once_with("*", "https://example.com/page")

    @patch('urllib.robotparser.RobotFileParser')
    @patch('time.sleep')
    def test_can_fetch_with_crawl_delay(self, mock_sleep, mock_parser_class):
        """Test can_fetch with crawl delay specified in robots.txt."""
        # Setup mock
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = True
        mock_parser.crawl_delay.return_value = 3.0  # 3 seconds crawl delay
        mock_parser_class.return_value = mock_parser

        # Create checker and test
        checker = RobotsChecker(rate_limit=1.0)
        result, reason = checker.can_fetch("https://example.com/page")

        # Assertions
        self.assertTrue(result)
        self.assertIsNone(reason)
        mock_parser.crawl_delay.assert_called_once_with("*")
        # Should use the larger of crawl_delay and rate_limit
        self.assertIn("https://example.com", checker.last_request_time)
        # Verify the timestamp is recent (within 1 second of now)
        self.assertLess(time.time() - checker.last_request_time["https://example.com"], 1.0)
        # First request shouldn't sleep
        mock_sleep.assert_not_called()

    @patch('urllib.robotparser.RobotFileParser')
    @patch('time.sleep')
    @patch('time.time')
    def test_can_fetch_rate_limiting(self, mock_time, mock_sleep, mock_parser_class):
        """Test rate limiting between requests."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = True
        mock_parser.crawl_delay.return_value = None
        mock_parser_class.return_value = mock_parser

        # Mock time.time to return controlled values
        mock_time.side_effect = [100.0, 100.5]  # First call and second call times

        # Create checker and set last request time manually
        checker = RobotsChecker(rate_limit=2.0)
        checker.parsers["https://example.com"] = mock_parser
        checker.last_request_time["https://example.com"] = 100.0  # Last request was at t=100

        # Test the method
        result, reason = checker.can_fetch("https://example.com/page")

        # Assertions
        self.assertTrue(result)
        self.assertIsNone(reason)
        # The code uses the rate_limit directly rather than calculating the difference
        mock_sleep.assert_called_once_with(2.0)

    @patch('urllib.robotparser.RobotFileParser')
    def test_can_fetch_exception_handling(self, mock_parser_class):
        """Test exception handling in can_fetch."""
        # Setup mock to raise an exception
        mock_parser = MagicMock()
        mock_parser.set_url.side_effect = Exception("Test exception")
        mock_parser_class.return_value = mock_parser

        # Create checker and test
        checker = RobotsChecker()
        with patch('time.sleep') as mock_sleep:
            result, reason = checker.can_fetch("https://example.com/page")

        # Assertions
        self.assertTrue(result)  # Should return True by default on error
        self.assertEqual(reason, "Erreur de vérification: Test exception")
        mock_sleep.assert_called_once_with(1.0)  # Should sleep for rate_limit

    @patch('urllib.robotparser.RobotFileParser')
    def test_can_fetch_reuses_parser(self, mock_parser_class):
        """Test that the parser is reused for the same domain."""
        # Setup mock
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = True
        mock_parser.crawl_delay.return_value = None
        mock_parser_class.return_value = mock_parser

        # Create checker and call can_fetch twice for the same domain
        checker = RobotsChecker()
        checker.can_fetch("https://example.com/page1")
        checker.can_fetch("https://example.com/page2")

        # Assertions
        self.assertEqual(len(checker.parsers), 1)
        mock_parser_class.assert_called_once()  # Parser should only be created once
        self.assertEqual(mock_parser.can_fetch.call_count, 2)  # But can_fetch should be called twice

if __name__ == '__main__':
    unittest.main()
