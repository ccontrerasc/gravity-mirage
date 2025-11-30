from pathlib import Path

from fastapi import HTTPException

from gravity_mirage.web.constants import EXPORT_FOLDER, UPLOAD_FOLDER


def resolve_uploaded_file(filename: str) -> Path:
    """Ensure the requested filename lives inside the uploads folder."""
    clean_name = Path(filename).name
    target = (UPLOAD_FOLDER / clean_name).resolve()
    base = UPLOAD_FOLDER.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return target


def resolve_export_file(filename: str) -> Path:
    """Ensure the requested filename lives inside the exports folder."""
    clean_name = Path(filename).name
    target = (EXPORT_FOLDER / clean_name).resolve()
    base = EXPORT_FOLDER.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return target
