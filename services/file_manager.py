import logging
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

def list_files(directory: str, subpath: str = ""):
    """List files and folders in a directory, optionally within a subpath"""
    base_dir = Path(directory)
    target_path = base_dir / subpath
    
    # Security check: ensure the user isn't trying to access files outside the folder
    try:
        target_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden: Path traversal not allowed")
    
    if not target_path.exists():
        if subpath:
             raise HTTPException(status_code=404, detail="Folder not found")
        return {"success": True, "items": [], "path": ""}
    
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    items = []
    for filepath in sorted(target_path.glob("*")):
        stat = filepath.stat()
        if filepath.is_dir():
            item_count = len(list(filepath.glob("*")))
            items.append({
                "name": filepath.name,
                "type": "folder",
                "item_count": item_count,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        else:
            items.append({
                "name": filepath.name,
                "type": "file",
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    items.sort(key=lambda x: x['modified'], reverse=True)
    return {"success": True, "items": items, "path": subpath}

def view_file(directory: str, filepath: str, user_email: str):
    """Serve a specific file securely from a directory"""
    base_dir = Path(directory)
    file_path = base_dir / filepath
    
    # Security check: ensure the user isn't trying to access files outside the folder
    try:
        file_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden: Path traversal not allowed")

    # Check if file exists
    if not file_path.exists():
        logger.warning(f"File not found: {file_path} (requested by {user_email})")
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    
    # Ensure it's a file, not a directory
    if file_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory, not a file")
    
    # Serve the file
    logger.info(f"Serving file {filepath} to {user_email}")
    return FileResponse(file_path)
