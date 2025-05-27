import unittest
import sys
import os

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processors.html_preprocessor import (
    preprocess_html,
    extract_main_content,
    get_page_title
)

class TestHtmlPreprocessor(unittest.TestCase):
    """Test cases for the HTML preprocessor module."""

    def test_preprocess_html_empty(self):
        """Test preprocessing with empty input."""
        self.assertEqual(preprocess_html(""), "")
        self.assertEqual(preprocess_html(None), "")

    def test_preprocess_html_basic(self):
        """Test basic HTML preprocessing."""
        html = """
        <html>
            <head>
                <title>Test Page</title>
                <style>body { color: red; }</style>
            </head>
            <body>
                <h1>Hello World</h1>
                <p>This is a test paragraph.</p>
                <script>alert('test');</script>
            </body>
        </html>
        """
        expected = "Hello World This is a test paragraph."
        self.assertEqual(preprocess_html(html).strip(), expected)

    def test_preprocess_html_comments(self):
        """Test that HTML comments are removed."""
        html = """
        <html>
            <body>
                <!-- This is a comment -->
                <p>This is visible text.</p>
                <!-- Another comment -->
            </body>
        </html>
        """
        expected = "This is visible text."
        self.assertEqual(preprocess_html(html).strip(), expected)

    def test_preprocess_html_special_chars(self):
        """Test handling of special characters."""
        html = """
        <html>
            <body>
                <p>Special characters: &lt; &gt; &amp; &quot; &apos;</p>
                <p>Currency symbols: € $ £ ¥</p>
            </body>
        </html>
        """
        # The actual output might differ slightly in spacing or exact character representation
        # Focus on checking that the key special characters are present
        result = preprocess_html(html).strip()
        self.assertIn("Special characters:", result)
        self.assertIn("Currency symbols:", result)
        self.assertIn("€", result)
        self.assertIn("$", result)
        self.assertIn("£", result)
        self.assertIn("¥", result)

    def test_preprocess_html_punctuation(self):
        """Test handling of punctuation."""
        html = """
        <html>
            <body>
                <p>This is a sentence . And another one !</p>
                <p>Comma , semicolon ; colon :</p>
            </body>
        </html>
        """
        expected = "This is a sentence. And another one! Comma, semicolon; colon:"
        self.assertEqual(preprocess_html(html).strip(), expected)

    def test_extract_main_content_empty(self):
        """Test main content extraction with empty input."""
        self.assertEqual(extract_main_content(""), "")
        self.assertEqual(extract_main_content(None), "")

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
                    <p>Article content.</p>
                </article>
                <footer>Footer</footer>
            </body>
        </html>
        """
        expected = "Article Title Article content."
        self.assertEqual(extract_main_content(html).strip(), expected)

    def test_extract_main_content_with_content_class(self):
        """Test extraction when there's a content class."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <header>Site Header</header>
                <div class="sidebar">Sidebar</div>
                <div class="content">
                    <h1>Main Content</h1>
                    <p>This is the main content.</p>
                </div>
                <footer>Footer</footer>
            </body>
        </html>
        """
        expected = "Main Content This is the main content."
        self.assertEqual(extract_main_content(html).strip(), expected)

    def test_extract_main_content_fallback(self):
        """Test extraction fallback to body when no main content is found."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <div>Some content without specific markers.</div>
                <p>More content.</p>
            </body>
        </html>
        """
        expected = "Some content without specific markers. More content."
        self.assertEqual(extract_main_content(html).strip(), expected)

    def test_get_page_title_empty(self):
        """Test title extraction with empty input."""
        self.assertEqual(get_page_title(""), "Pas de titre trouvé")
        self.assertEqual(get_page_title(None), "Pas de titre trouvé")

    def test_get_page_title_basic(self):
        """Test basic title extraction."""
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

    def test_get_page_title_missing(self):
        """Test title extraction when title tag is missing."""
        html = """
        <html>
            <head>
                <meta name="description" content="Test">
            </head>
            <body>
                <h1>Heading</h1>
            </body>
        </html>
        """
        self.assertEqual(get_page_title(html), "Pas de titre trouvé")

    def test_get_page_title_malformed(self):
        """Test title extraction with malformed HTML."""
        html = "<html><head><title>Incomplete"
        self.assertEqual(get_page_title(html), "Incomplete")

        html = "<not-valid-html>"
        self.assertEqual(get_page_title(html), "Pas de titre trouvé")

if __name__ == '__main__':
    unittest.main()
