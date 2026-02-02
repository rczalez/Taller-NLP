from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from docx import Document


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts).strip()


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_documents(docs_dir: str = "docs") -> List[Tuple[str, str]]:
    """
    Returns list of (source_name, text)
    """
    p = Path(docs_dir)
    p.mkdir(parents=True, exist_ok=True)

    docs: List[Tuple[str, str]] = []
    for file in sorted(p.glob("*")):
        if file.is_dir():
            continue
        suffix = file.suffix.lower()
        try:
            if suffix == ".pdf":
                text = read_pdf(file)
            elif suffix == ".docx":
                text = read_docx(file)
            elif suffix in [".txt", ".md"]:
                text = read_txt(file)
            else:
                continue
            if text.strip():
                docs.append((file.name, text))
        except Exception:
            # Skip problematic docs in a beginner-friendly way
            continue
    return docs
