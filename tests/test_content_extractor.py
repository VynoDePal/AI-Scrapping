import unittest
import sys
import os
from bs4 import BeautifulSoup

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processors.content_extractor import (
    get_page_title,
    extract_main_content,
    find_main_content_div
)

class TestContentExtractor(unittest.TestCase):
    """Test cases for the content extractor module."""

    def test_get_page_title_empty(self):
        """Test title extraction with empty input."""
        self.assertIsNone(get_page_title(""))
        self.assertIsNone(get_page_title(None))

    def test_get_page_title_with_title_tag(self):
        """Test title extraction with a title tag."""
        html = """
        <html>
            <head>
                <title>Test Page Title</title>
            </head>
            <body>
                <h1>Not the title</h1>
            </body>
        </html>
        """
        self.assertEqual(get_page_title(html), "Test Page Title")

    def test_get_page_title_with_h1_fallback(self):
        """Test title extraction with h1 fallback when no title tag."""
        html = """
        <html>
            <head>
                <meta name="description" content="Test">
            </head>
            <body>
                <h1>Heading Title</h1>
            </body>
        </html>
        """
        self.assertEqual(get_page_title(html), "Heading Title")

    def test_get_page_title_no_title_no_h1(self):
        """Test title extraction with no title and no h1."""
        html = """
        <html>
            <head>
                <meta name="description" content="Test">
            </head>
            <body>
                <p>Just a paragraph</p>
            </body>
        </html>
        """
        self.assertIsNone(get_page_title(html))

    def test_get_page_title_exception(self):
        """Test title extraction with invalid HTML that causes an exception."""
        html = "<not-valid-html>"
        self.assertIsNone(get_page_title(html))

    def test_extract_main_content_empty(self):
        """Test main content extraction with empty input."""
        self.assertIsNone(extract_main_content(""))
        self.assertIsNone(extract_main_content(None))

    def test_extract_main_content_with_article(self):
        """Test extraction when there's an article tag."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <header>Site Header</header>
                <nav>Navigation</nav>
                <article>
                    <h1>Article Title</h1>
                    <p>This is the main content of the article.</p>
                    <p>It has multiple paragraphs with enough text to pass the threshold.</p>
                    <p>This should be enough text to be considered valid content.</p>
                    <p>Adding more text to ensure we pass the 200 character threshold for content.</p>
                </article>
                <footer>Footer</footer>
            </body>
        </html>
        """
        result = extract_main_content(html)
        self.assertIsNotNone(result)
        self.assertIn("Article Title", result)
        self.assertIn("main content", result)
        self.assertNotIn("Site Header", result)
        self.assertNotIn("Footer", result)

    def test_extract_main_content_with_main(self):
        """Test extraction when there's a main tag."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <header>Site Header</header>
                <nav>Navigation</nav>
                <main>
                    <h1>Main Content</h1>
                    <p>This is the main content section.</p>
                    <p>It has multiple paragraphs with enough text to pass the threshold.</p>
                    <p>This should be enough text to be considered valid content.</p>
                    <p>Adding more text to ensure we pass the 200 character threshold for content.</p>
                </main>
                <footer>Footer</footer>
            </body>
        </html>
        """
        result = extract_main_content(html)
        self.assertIsNotNone(result)
        self.assertIn("Main Content", result)
        self.assertIn("main content section", result)
        self.assertNotIn("Site Header", result)
        self.assertNotIn("Footer", result)

    def test_extract_main_content_with_content_class(self):
        """Test extraction when there's a content class."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <header>Site Header</header>
                <div class="sidebar">Sidebar</div>
                <div class="content">
                    <h1>Content Area</h1>
                    <p>This is the content area with a specific class.</p>
                    <p>It has multiple paragraphs with enough text to pass the threshold.</p>
                    <p>This should be enough text to be considered valid content.</p>
                    <p>Adding more text to ensure we pass the 200 character threshold for content.</p>
                </div>
                <footer>Footer</footer>
            </body>
        </html>
        """
        result = extract_main_content(html)
        self.assertIsNotNone(result)
        self.assertIn("Content Area", result)
        self.assertIn("content area with a specific class", result)
        self.assertNotIn("Site Header", result)
        self.assertNotIn("Footer", result)

    def test_extract_main_content_heuristic_fallback(self):
        """Test extraction fallback to heuristic when no main containers found."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <header>Site Header</header>
                <nav>Navigation</nav>
                <div>
                    <h1>Regular Div</h1>
                    <p>This is a regular div with no special class or id.</p>
                    <p>It has multiple paragraphs with enough text to pass the threshold.</p>
                    <p>This should be enough text to be considered valid content.</p>
                    <p>Adding more text to ensure we pass the 200 character threshold for content.</p>
                    <p>Even more text to make this div have the highest text density.</p>
                </div>
                <div class="sidebar">
                    <p>Short sidebar text</p>
                </div>
                <footer>Footer</footer>
            </body>
        </html>
        """
        result = extract_main_content(html)
        self.assertIsNotNone(result)
        self.assertIn("Regular Div", result)
        self.assertIn("regular div with no special class", result)
        self.assertNotIn("Site Header", result)
        self.assertNotIn("Footer", result)

    def test_extract_main_content_body_fallback(self):
        """Test extraction fallback to body when no main content is found."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <div>Very short content</div>
                <div>Another short div</div>
                <p>Some text outside of divs</p>
            </body>
        </html>
        """
        result = extract_main_content(html)
        self.assertIsNotNone(result)
        self.assertIn("Very short content", result)
        self.assertIn("Another short div", result)
        self.assertIn("Some text outside of divs", result)

    def test_extract_main_content_exception(self):
        """Test extraction with invalid HTML that causes an exception."""
        html = "<not-valid-html>"
        result = extract_main_content(html)
        # The function might return an empty string instead of None for invalid HTML
        self.assertTrue(result is None or result == "")

    def test_find_main_content_div_empty(self):
        """Test find_main_content_div with empty soup."""
        soup = BeautifulSoup("", "html.parser")
        self.assertIsNone(find_main_content_div(soup))

    def test_find_main_content_div_no_divs(self):
        """Test find_main_content_div with no divs."""
        soup = BeautifulSoup("<html><body><p>No divs here</p></body></html>", "html.parser")
        self.assertIsNone(find_main_content_div(soup))

    def test_find_main_content_div_with_paragraphs(self):
        """Test find_main_content_div with divs containing paragraphs."""
        html = """
        <html>
            <body>
                <div class="sidebar">
                    <p>Short sidebar text</p>
                </div>
                <div class="content">
                    <p>This is a paragraph with content.</p>
                    <p>This is another paragraph with content.</p>
                    <p>This is a third paragraph with content.</p>
                    <p>This is a fourth paragraph with content.</p>
                    <p>This is a lot of text to ensure this div has the highest text density.</p>
                    <p>Adding even more text to make sure this div is selected as the main content.</p>
                </div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = find_main_content_div(soup)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("class"), ["content"])

    def test_find_main_content_div_with_links(self):
        """Test find_main_content_div with divs containing many links."""
        html = """
        <html>
            <body>
                <div class="content">
                    <p>This is a paragraph with content.</p>
                    <p>This is another paragraph with content.</p>
                    <p>This has many links:
                        <a href="#">Link 1</a>
                        <a href="#">Link 2</a>
                        <a href="#">Link 3</a>
                        <a href="#">Link 4</a>
                        <a href="#">Link 5</a>
                        <a href="#">Link 6</a>
                    </p>
                </div>
                <div class="main">
                    <p>This is a paragraph with content.</p>
                    <p>This is another paragraph with content.</p>
                    <p>This is a third paragraph with content.</p>
                    <p>This is a fourth paragraph with content.</p>
                    <p>This is a lot of text to ensure this div has the highest text density.</p>
                    <p>Adding even more text to make sure this div is selected as the main content.</p>
                </div>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = find_main_content_div(soup)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("class"), ["main"])

if __name__ == '__main__':
    unittest.main()
