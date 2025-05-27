import unittest
import sys
import os
from bs4 import BeautifulSoup

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processors.html_chunker import (
    chunk_by_tags,
    chunk_by_length,
    html_to_chunks
)

class TestHtmlChunker(unittest.TestCase):
    """Test cases for the HTML chunker module."""

    def test_chunk_by_length_empty(self):
        """Test chunking by length with empty input."""
        self.assertEqual(chunk_by_length(""), [])
        self.assertEqual(chunk_by_length(None), [])

    def test_chunk_by_length_short_text(self):
        """Test chunking by length with text shorter than max_length."""
        text = "This is a short text."
        self.assertEqual(chunk_by_length(text, max_length=100), [text])

    def test_chunk_by_length_with_sentence_breaks(self):
        """Test chunking by length with natural sentence breaks."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        chunks = chunk_by_length(text, max_length=30, overlap=5)
        # The actual number of chunks depends on the implementation details
        self.assertTrue(len(chunks) >= 2)
        # Check that the first chunk contains the first sentence
        self.assertIn("first sentence", chunks[0])
        # Check that a later chunk contains the second sentence
        self.assertTrue(any("second sentence" in chunk for chunk in chunks))

    def test_chunk_by_length_with_newlines(self):
        """Test chunking by length with newline breaks."""
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunk_by_length(text, max_length=12, overlap=2)
        self.assertTrue(len(chunks) > 1)
        # The function finds '\n' as a separator, so the first chunk ends after "Line 1\n"
        self.assertEqual(chunks[0], "Line 1\n")

    def test_chunk_by_length_no_good_breaks(self):
        """Test chunking by length when no good break points are found."""
        text = "ThisIsAVeryLongWordWithoutAnyGoodBreakPoints" * 5
        chunks = chunk_by_length(text, max_length=50, overlap=10)
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(len(chunks[0]), 50)  # Should be exactly max_length

    def test_chunk_by_length_with_overlap(self):
        """Test that chunking by length properly handles overlap."""
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        chunks = chunk_by_length(text, max_length=20, overlap=5)
        self.assertTrue(len(chunks) > 1)
        # Check that the second chunk starts with text from the end of the first chunk
        self.assertTrue(chunks[1].startswith(chunks[0][-5:]) or 
                       any(word in chunks[1].split() for word in chunks[0][-5:].split()))

    def test_chunk_by_tags_empty(self):
        """Test chunking by tags with empty input."""
        soup = BeautifulSoup("", "html.parser")
        self.assertEqual(chunk_by_tags(soup), [])

    def test_chunk_by_tags_no_tags(self):
        """Test chunking by tags when no matching tags are found."""
        html = "<html><body>Plain text without any paragraph tags</body></html>"
        soup = BeautifulSoup(html, "html.parser")
        chunks = chunk_by_tags(soup)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Plain text", chunks[0])

    def test_chunk_by_tags_with_paragraphs(self):
        """Test chunking by tags with paragraph tags."""
        html = """
        <html>
            <body>
                <p>Paragraph 1</p>
                <p>Paragraph 2</p>
                <p>Paragraph 3</p>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        chunks = chunk_by_tags(soup)
        self.assertEqual(len(chunks), 1)  # Should combine into one chunk since it's short
        self.assertIn("Paragraph 1", chunks[0])
        self.assertIn("Paragraph 2", chunks[0])
        self.assertIn("Paragraph 3", chunks[0])

    def test_chunk_by_tags_with_headings(self):
        """Test chunking by tags with heading context."""
        html = """
        <html>
            <body>
                <h1>Main Title</h1>
                <p>Content under main title.</p>
                <h2>Subtitle</h2>
                <p>Content under subtitle.</p>
                <h3>Sub-subtitle</h3>
                <p>Content under sub-subtitle.</p>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        chunks = chunk_by_tags(soup)
        self.assertTrue(len(chunks) >= 1)
        # Check that heading context is preserved
        self.assertIn("Main Title", chunks[0])
        self.assertIn("Content under main title", chunks[0])

    def test_chunk_by_tags_long_element(self):
        """Test chunking by tags with an element that exceeds max_length."""
        long_text = "This is a very long paragraph. " * 50  # Will exceed default max_length
        html = f"<html><body><p>{long_text}</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        chunks = chunk_by_tags(soup, max_length=200)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))

    def test_chunk_by_tags_exception_handling(self):
        """Test that chunk_by_tags handles exceptions gracefully."""
        # Create a mock soup that will cause an exception when find_all is called
        mock_soup = type('MockSoup', (), {
            'find_all': lambda self, *args, **kwargs: 1/0,  # Will raise ZeroDivisionError
            'get_text': lambda self, separator: "Fallback text"
        })()

        # Should fall back to chunk_by_length
        chunks = chunk_by_tags(mock_soup, max_length=100)
        self.assertEqual(chunks, ["Fallback text"])

    def test_html_to_chunks_empty(self):
        """Test html_to_chunks with empty input."""
        self.assertEqual(html_to_chunks(""), [])
        self.assertEqual(html_to_chunks(None), [])

    def test_html_to_chunks_tags_method(self):
        """Test html_to_chunks with 'tags' method."""
        html = """
        <html>
            <body>
                <p>Paragraph 1</p>
                <p>Paragraph 2</p>
            </body>
        </html>
        """
        chunks = html_to_chunks(html, method='tags')
        self.assertTrue(len(chunks) >= 1)
        self.assertIn("Paragraph 1", chunks[0])

    def test_html_to_chunks_length_method(self):
        """Test html_to_chunks with 'length' method."""
        html = """
        <html>
            <body>
                <p>Paragraph 1</p>
                <p>Paragraph 2</p>
            </body>
        </html>
        """
        chunks = html_to_chunks(html, method='length')
        self.assertTrue(len(chunks) >= 1)
        self.assertIn("Paragraph 1", chunks[0])

    def test_html_to_chunks_hybrid_method(self):
        """Test html_to_chunks with 'hybrid' method."""
        html = """
        <html>
            <body>
                <p>Paragraph 1</p>
                <p>Paragraph 2</p>
            </body>
        </html>
        """
        chunks = html_to_chunks(html, method='hybrid')
        self.assertTrue(len(chunks) >= 1)
        self.assertIn("Paragraph 1", chunks[0])

    def test_html_to_chunks_with_script_removal(self):
        """Test that html_to_chunks removes script and style tags."""
        html = """
        <html>
            <head>
                <script>alert('This should be removed');</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <p>This content should remain.</p>
            </body>
        </html>
        """
        chunks = html_to_chunks(html)
        self.assertTrue(len(chunks) >= 1)
        self.assertIn("This content should remain", chunks[0])
        self.assertNotIn("alert", chunks[0])
        self.assertNotIn("color: red", chunks[0])

    def test_html_to_chunks_exception_handling(self):
        """Test that html_to_chunks handles exceptions gracefully."""
        # Invalid HTML that will cause BeautifulSoup to struggle
        invalid_html = "<html><unclosed_tag>This is broken HTML</html>"
        chunks = html_to_chunks(invalid_html)
        self.assertTrue(len(chunks) >= 1)
        self.assertIn("This is broken HTML", chunks[0])

    def test_html_to_chunks_with_long_content(self):
        """Test html_to_chunks with content that needs to be split."""
        long_text = "This is a sentence. " * 100  # Will exceed default max_length
        html = f"<html><body><p>{long_text}</p></body></html>"
        chunks = html_to_chunks(html, max_length=200)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))

if __name__ == '__main__':
    unittest.main()
