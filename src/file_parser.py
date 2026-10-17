"""
File Parser Module for AI Meeting Intelligence.
Extracts raw text content from .txt, .pdf, and .docx files.
"""

import os
import io

def parse_txt(file_bytes_or_path):
    """Parse UTF-8 text from bytes or path."""
    if isinstance(file_bytes_or_path, (str, os.PathLike)):
        with open(file_bytes_or_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif isinstance(file_bytes_or_path, bytes):
        return file_bytes_or_path.decode("utf-8", errors="ignore")
    elif hasattr(file_bytes_or_path, "read"):
        content = file_bytes_or_path.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
    return str(file_bytes_or_path)

def parse_pdf(file_bytes_or_path):
    """Extract text from PDF file using pypdf/PyPDF2 or pdfplumber if available."""
    text = ""
    try:
        import pypdf
        if isinstance(file_bytes_or_path, (str, os.PathLike)):
            reader = pypdf.PdfReader(file_bytes_or_path)
        else:
            if hasattr(file_bytes_or_path, "getvalue"):
                stream = io.BytesIO(file_bytes_or_path.getvalue())
            elif isinstance(file_bytes_or_path, bytes):
                stream = io.BytesIO(file_bytes_or_path)
            else:
                stream = file_bytes_or_path
            reader = pypdf.PdfReader(stream)
            
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        pass

    try:
        import PyPDF2
        if isinstance(file_bytes_or_path, (str, os.PathLike)):
            reader = PyPDF2.PdfReader(file_bytes_or_path)
        else:
            if hasattr(file_bytes_or_path, "getvalue"):
                stream = io.BytesIO(file_bytes_or_path.getvalue())
            elif isinstance(file_bytes_or_path, bytes):
                stream = io.BytesIO(file_bytes_or_path)
            else:
                stream = file_bytes_or_path
            reader = PyPDF2.PdfReader(stream)
            
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        pass

    # Fallback return text if found or raw string decoding
    if not text and isinstance(file_bytes_or_path, bytes):
        return file_bytes_or_path.decode("utf-8", errors="ignore")
    return text

def parse_docx(file_bytes_or_path):
    """Extract text from DOCX file using python-docx."""
    try:
        import docx
        if isinstance(file_bytes_or_path, (str, os.PathLike)):
            doc = docx.Document(file_bytes_or_path)
        else:
            if hasattr(file_bytes_or_path, "getvalue"):
                stream = io.BytesIO(file_bytes_or_path.getvalue())
            elif isinstance(file_bytes_or_path, bytes):
                stream = io.BytesIO(file_bytes_or_path)
            else:
                stream = file_bytes_or_path
            doc = docx.Document(stream)
            
        full_text = []
        for para in doc.paragraphs:
            if para.text:
                full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        if isinstance(file_bytes_or_path, bytes):
            return file_bytes_or_path.decode("utf-8", errors="ignore")
        return str(file_bytes_or_path)

def extract_text_from_file(file_obj, filename):
    """
    Main file parsing dispatcher based on file extension.
    Supports .txt, .pdf, and .docx.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return parse_txt(file_obj)
    elif ext == ".pdf":
        return parse_pdf(file_obj)
    elif ext in [".docx", ".doc"]:
        return parse_docx(file_obj)
    else:
        # Default text reading fallback
        return parse_txt(file_obj)
