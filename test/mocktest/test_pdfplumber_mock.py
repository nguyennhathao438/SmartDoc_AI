# test/mocktest/test_pdfplumber_mock_pytest.py
import pytest
from unittest.mock import MagicMock, patch


class TestPdfplumberMock:
    """Test mock cho pdfplumber sử dụng pytest"""
    
    @patch('rag.loader.pdfplumber')
    def test_extract_pages_success(self, mock_pdfplumber):
        """Test extract PDF thành công"""
        
        # Mock setup
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test content"
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        # Import và test
        from rag.loader import extract_pages_from_pdf
        result = extract_pages_from_pdf("test.pdf")
        
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert result[0]["text"] == "Test content"
        mock_pdfplumber.open.assert_called_once_with("test.pdf")
    
    @patch('rag.loader.pdfplumber')
    def test_extract_pages_multiple(self, mock_pdfplumber):
        """Test extract PDF nhiều trang"""
        
        # Mock 3 pages
        mock_pages = []
        for i in range(3):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = f"Page {i+1} content"
            mock_pages.append(mock_page)
        
        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__.return_value = mock_pdf
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        from rag.loader import extract_pages_from_pdf
        result = extract_pages_from_pdf("multi.pdf")
        
        assert len(result) == 3
        assert [p["page"] for p in result] == [1, 2, 3]
    
    @patch('rag.loader.pdfplumber')
    def test_extract_pages_empty_page(self, mock_pdfplumber):
        """Test trang PDF rỗng"""
        
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = None  # Empty page
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Real content"
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__.return_value = mock_pdf
        
        mock_pdfplumber.open.return_value = mock_pdf
        
        from rag.loader import extract_pages_from_pdf
        result = extract_pages_from_pdf("empty.pdf")
        
        # Only page with content is included
        assert len(result) == 1
        assert result[0]["page"] == 2
    
    @patch('rag.loader.pdfplumber')
    def test_extract_pages_file_not_found(self, mock_pdfplumber):
        """Test file không tồn tại"""
        
        mock_pdfplumber.open.side_effect = FileNotFoundError("No such file")
        
        from rag.loader import extract_pages_from_pdf
        
        with pytest.raises(FileNotFoundError):
            extract_pages_from_pdf("missing.pdf")