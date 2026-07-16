try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except Exception:
    PdfReader = None
    _PYPDF_AVAILABLE = False


def extract_text_from_upload(uploaded):
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    data = uploaded.read()
    try:
        if name.endswith('.pdf'):
            if not _PYPDF_AVAILABLE:
                return ""  # indicate unavailable PDF parsing
            # pypdf can accept a file-like object
            reader = PdfReader(uploaded)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages)
        else:
            return data.decode('utf-8', errors='ignore')
    except Exception:
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return ""
