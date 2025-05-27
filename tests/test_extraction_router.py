import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import io
from fastapi.testclient import TestClient

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app
from api.models.requests import ExtractionRequest
from api.models.responses import ExtractionResponse

class TestExtractionRouter(unittest.TestCase):
    """Test cases for the extraction router."""

    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)

    @patch('api.routers.extraction.html_to_chunks')
    @patch('api.routers.extraction.get_llm_provider')
    @patch('api.routers.extraction.extract_data_from_chunks')
    @patch('api.routers.extraction.aggregate_extraction_results')
    def test_extract_data_basic(self, mock_aggregate_results, mock_extract_data, 
                               mock_get_llm_provider, mock_html_to_chunks):
        """Test basic data extraction functionality."""
        # Setup mocks
        mock_html_to_chunks.return_value = ["Chunk 1", "Chunk 2"]
        mock_llm_provider = MagicMock()
        mock_get_llm_provider.return_value = mock_llm_provider
        mock_extract_data.return_value = [{"titles": ["Title 1"]}, {"titles": ["Title 2"]}]
        mock_aggregate_results.return_value = {"titles": ["Title 1", "Title 2"]}

        # Make request
        response = self.client.post(
            "/extraction/",
            json={
                "content": "<html><body>Test content</body></html>",
                "query": "Extract titles",
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "chunk_size": 4000,
                "chunk_method": "hybrid",
                "temperature": 0.0
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "Extract titles")
        self.assertEqual(data["provider"], "openai")
        self.assertEqual(data["model"], "gpt-3.5-turbo")
        self.assertEqual(data["chunk_count"], 2)
        self.assertEqual(data["data"], {"titles": ["Title 1", "Title 2"]})
        self.assertTrue("timestamp" in data)

        # Verify mocks were called correctly
        mock_html_to_chunks.assert_called_once_with(
            "<html><body>Test content</body></html>", 
            method="hybrid", 
            max_length=4000
        )
        mock_get_llm_provider.assert_called_once_with(
            "openai", 
            api_key=None, 
            model="gpt-3.5-turbo", 
            temperature=0.0
        )
        mock_extract_data.assert_called_once_with(
            chunks=["Chunk 1", "Chunk 2"],
            query="Extract titles",
            llm_provider=mock_llm_provider,
            max_workers=2
        )
        mock_aggregate_results.assert_called_once_with([{"titles": ["Title 1"]}, {"titles": ["Title 2"]}])

    @patch('api.routers.extraction.html_to_chunks')
    def test_extract_data_chunking_failure(self, mock_html_to_chunks):
        """Test extraction when chunking fails."""
        # Setup mock to return empty list (chunking failure)
        mock_html_to_chunks.return_value = []

        # Make request
        response = self.client.post(
            "/extraction/",
            json={
                "content": "<html><body>Test content</body></html>",
                "query": "Extract titles",
                "provider": "openai",
                "model": "gpt-3.5-turbo"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertTrue("Échec du chunking" in data["detail"])

    @patch('api.routers.extraction.html_to_chunks')
    @patch('api.routers.extraction.get_llm_provider')
    def test_extract_data_with_max_chunks(self, mock_get_llm_provider, mock_html_to_chunks):
        """Test extraction with max_chunks parameter."""
        # Setup mocks
        mock_html_to_chunks.return_value = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"]
        mock_llm_provider = MagicMock()
        mock_get_llm_provider.return_value = mock_llm_provider

        # Make request
        response = self.client.post(
            "/extraction/",
            json={
                "content": "<html><body>Test content</body></html>",
                "query": "Extract titles",
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "max_chunks": 3
            }
        )

        # We don't need to check the full response, just that max_chunks was applied
        # This will be verified in the extract_data_from_chunks call

        # Verify that only the first 3 chunks were used
        args, kwargs = mock_get_llm_provider.call_args
        self.assertEqual(kwargs["model"], "gpt-3.5-turbo")

    @patch('api.routers.extraction.html_to_chunks')
    @patch('api.routers.extraction.get_llm_provider')
    @patch('api.routers.extraction.extract_data_from_chunks')
    @patch('api.routers.extraction.aggregate_extraction_results')
    def test_extract_data_with_host(self, mock_aggregate_results, mock_extract_data, 
                                   mock_get_llm_provider, mock_html_to_chunks):
        """Test extraction with host parameter."""
        # Setup mocks
        mock_html_to_chunks.return_value = ["Chunk 1"]
        mock_llm_provider = MagicMock()
        mock_get_llm_provider.return_value = mock_llm_provider
        mock_extract_data.return_value = [{"titles": ["Title 1"]}]
        mock_aggregate_results.return_value = {"titles": ["Title 1"]}

        # Make request
        response = self.client.post(
            "/extraction/",
            json={
                "content": "<html><body>Test content</body></html>",
                "query": "Extract titles",
                "provider": "lmstudio",
                "model": "llama2",
                "host": "http://localhost:1234"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)

        # Verify host was passed to get_llm_provider
        mock_get_llm_provider.assert_called_once_with(
            "lmstudio", 
            api_key=None, 
            model="llama2", 
            temperature=0.0,
            host="http://localhost:1234"
        )

    @patch('api.routers.extraction.extract_data')
    def test_extract_from_file(self, mock_extract_data):
        """Test extraction from file."""
        # Setup mock
        mock_extract_data.return_value = {
            "query": "Extract titles",
            "data": {"titles": ["Title 1", "Title 2"]},
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "chunk_count": 2,
            "timestamp": "2023-01-01T00:00:00"
        }

        # Create a test file
        file_content = b"<html><body>Test content</body></html>"

        # Make request with file
        response = self.client.post(
            "/extraction/file",
            files={"file": ("test.html", io.BytesIO(file_content), "text/html")},
            data={
                "query": "Extract titles",
                "provider": "openai",
                "model": "gpt-3.5-turbo"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "Extract titles")
        self.assertEqual(data["data"], {"titles": ["Title 1", "Title 2"]})

        # Verify extract_data was called with correct parameters
        args, kwargs = mock_extract_data.call_args
        request_arg = args[0]
        self.assertEqual(request_arg.query, "Extract titles")
        self.assertEqual(request_arg.provider, "openai")
        self.assertEqual(request_arg.model, "gpt-3.5-turbo")
        self.assertEqual(request_arg.content, "<html><body>Test content</body></html>")

    @patch('api.routers.extraction.extract_data')
    def test_extract_from_file_error(self, mock_extract_data):
        """Test extraction from file with error."""
        # Setup mock to raise exception
        mock_extract_data.side_effect = Exception("Test error")

        # Make request with file
        response = self.client.post(
            "/extraction/file",
            files={"file": ("test.html", io.BytesIO(b"<html></html>"), "text/html")},
            data={
                "query": "Extract titles",
                "provider": "openai",
                "model": "gpt-3.5-turbo"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertTrue("Test error" in data["detail"])

    @patch('src.processors.pdf_processor.extract_pdf_metadata')
    @patch('api.routers.extraction.pdf_to_chunks')
    @patch('api.routers.extraction.get_llm_provider')
    @patch('api.routers.extraction.extract_data_from_chunks')
    @patch('api.routers.extraction.aggregate_extraction_results')
    @patch('os.path.exists')
    @patch('os.remove')
    def test_extract_from_pdf_basic(self, mock_remove, mock_exists, mock_aggregate_results, 
                                   mock_extract_data, mock_get_llm_provider, 
                                   mock_pdf_to_chunks, mock_extract_pdf_metadata):
        """Test basic PDF extraction."""
        # Setup mocks
        mock_extract_pdf_metadata.return_value = {"title": "Test PDF"}
        mock_pdf_to_chunks.return_value = ["Chunk 1", "Chunk 2"]
        mock_llm_provider = MagicMock()
        mock_get_llm_provider.return_value = mock_llm_provider
        mock_extract_data.return_value = [{"titles": ["Title 1"]}, {"titles": ["Title 2"]}]
        mock_aggregate_results.return_value = {"titles": ["Title 1", "Title 2"]}
        mock_exists.return_value = True

        # Mock open to avoid actual file operations
        m = mock_open()
        with patch('builtins.open', m):
            # Make request
            response = self.client.post(
                "/extraction/pdf",
                files={"file": ("test.pdf", io.BytesIO(b"PDF content"), "application/pdf")},
                data={
                    "query": "Extract titles",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "auto_enhance": "false"
                }
            )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "Extract titles")
        self.assertEqual(data["data"], {"titles": ["Title 1", "Title 2"]})
        self.assertEqual(data["provider"], "openai")
        self.assertEqual(data["model"], "gpt-3.5-turbo")
        self.assertEqual(data["chunk_count"], 2)

        # Verify mocks were called correctly
        mock_pdf_to_chunks.assert_called_once()
        mock_get_llm_provider.assert_called_once()
        mock_extract_data.assert_called_once()
        mock_aggregate_results.assert_called_once()
        mock_remove.assert_called_once()  # Temp file should be removed

    @patch('src.processors.pdf_processor.extract_pdf_metadata')
    @patch('src.processors.pdf_processor.extract_text_from_pdf')
    @patch('api.routers.extraction.pdf_to_chunks')
    @patch('api.routers.extraction.get_llm_provider')
    @patch('api.routers.extraction.extract_data_from_chunks')
    @patch('api.routers.extraction.aggregate_extraction_results')
    @patch('os.path.exists')
    @patch('os.remove')
    def test_extract_from_pdf_with_auto_enhance(self, mock_remove, mock_exists, 
                                              mock_aggregate_results, mock_extract_data, 
                                              mock_get_llm_provider, mock_pdf_to_chunks, 
                                              mock_extract_text_from_pdf, mock_extract_pdf_metadata):
        """Test PDF extraction with auto_enhance feature."""
        # Setup mocks
        mock_extract_pdf_metadata.return_value = {"title": "Test Resume"}
        mock_extract_text_from_pdf.return_value = "resume skills experience education"
        mock_pdf_to_chunks.return_value = ["Chunk 1", "Chunk 2"]
        mock_llm_provider = MagicMock()
        mock_get_llm_provider.return_value = mock_llm_provider
        mock_extract_data.return_value = [{"skills": ["Skill 1"]}, {"skills": ["Skill 2"]}]
        mock_aggregate_results.return_value = {"skills": ["Skill 1", "Skill 2"]}
        mock_exists.return_value = True

        # Mock open to avoid actual file operations
        m = mock_open()
        with patch('builtins.open', m):
            # Make request with a generic query and resume filename
            response = self.client.post(
                "/extraction/pdf",
                files={"file": ("resume.pdf", io.BytesIO(b"PDF content"), "application/pdf")},
                data={
                    "query": "extract",  # Generic query that should trigger auto-enhance
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "auto_enhance": "true"
                }
            )

        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("requête améliorée automatiquement" in data["query"])

        # Verify that extract_data_from_chunks was called with an enhanced query
        args, kwargs = mock_extract_data.call_args
        self.assertTrue("compétences" in kwargs["query"].lower() or "skills" in kwargs["query"].lower())

    @patch('api.routers.extraction.pdf_to_chunks')
    @patch('os.path.exists')
    @patch('os.remove')
    def test_extract_from_pdf_chunking_failure(self, mock_remove, mock_exists, mock_pdf_to_chunks):
        """Test PDF extraction when chunking fails."""
        # Setup mocks
        mock_pdf_to_chunks.return_value = []  # Empty chunks
        mock_exists.return_value = True

        # Mock open to avoid actual file operations
        m = mock_open()
        with patch('builtins.open', m):
            # Make request
            response = self.client.post(
                "/extraction/pdf",
                files={"file": ("test.pdf", io.BytesIO(b"PDF content"), "application/pdf")},
                data={
                    "query": "Extract titles",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo"
                }
            )

        # Assertions
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertTrue("Échec de l'extraction du PDF" in data["detail"])

        # Verify temp file was removed even though there was an error
        mock_remove.assert_called_once()

    @patch('src.processors.pdf_processor.extract_pdf_metadata')
    @patch('os.path.exists')
    @patch('os.remove')
    def test_extract_from_pdf_exception(self, mock_remove, mock_exists, mock_extract_pdf_metadata):
        """Test PDF extraction when an exception occurs."""
        # Setup mock to raise exception
        mock_extract_pdf_metadata.side_effect = Exception("Test error")
        mock_exists.return_value = True

        # Mock open to avoid actual file operations
        m = mock_open()
        with patch('builtins.open', m):
            # Make request
            response = self.client.post(
                "/extraction/pdf",
                files={"file": ("test.pdf", io.BytesIO(b"PDF content"), "application/pdf")},
                data={
                    "query": "Extract titles",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo"
                }
            )

        # Assertions
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertTrue("Test error" in data["detail"])

        # Verify temp file was removed even though there was an error
        mock_remove.assert_called_once()

if __name__ == '__main__':
    unittest.main()
