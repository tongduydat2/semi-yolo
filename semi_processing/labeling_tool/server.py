"""
FastAPI server for web-based YOLO annotation tool.

Provides REST API for:
- Serving images from eval directory
- Reading/writing labels in YOLO format
- Class configuration

Usage:
    python -m labeling_tool.server --images /path/to/images --port 8765
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

# ============================================================================
# Helper Functions
# ============================================================================

# Pre-defined color palette for classes
CLASS_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FF4500",
]

def load_classes_from_yaml(yaml_path: str) -> tuple[Dict[int, Dict[str, str]], Optional[str]]:
    """Load class names and val path from YOLO data.yaml file.
    
    Args:
        yaml_path: Path to data.yaml file
        
    Returns:
        Tuple of (classes dict, val_path or None)
        classes: Dict mapping class_id -> {"name": str, "color": str}
        val_path: Path to validation images (or None if not specified)
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    classes = {}
    names = data.get('names', {})
    
    # Handle both dict format {0: "name"} and list format ["name0", "name1"]
    if isinstance(names, dict):
        for class_id, name in names.items():
            classes[int(class_id)] = {
                "name": str(name),
                "color": CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]
            }
    elif isinstance(names, list):
        for class_id, name in enumerate(names):
            classes[class_id] = {
                "name": str(name),
                "color": CLASS_COLORS[class_id % len(CLASS_COLORS)]
            }
    
    # Get val path
    val_path = data.get('val', None)
    
    return classes, val_path

# ============================================================================
# Configuration
# ============================================================================

# Default class configuration (can be overridden via --config)
DEFAULT_CLASSES = {
    0: {"name": "09hot cell", "color": "#FF6B6B"},
    1: {"name": "07break", "color": "#4ECDC4"},
    2: {"name": "05shadow", "color": "#45B7D1"},
    3: {"name": "03string short circuit", "color": "#96CEB4"},
    4: {"name": "01substring open circuit", "color": "#FFEAA7"},
}

# ============================================================================
# Pydantic Models
# ============================================================================

class BoundingBox(BaseModel):
    """Single bounding box annotation in YOLO format."""
    class_id: int
    cx: float  # Center x (normalized 0-1)
    cy: float  # Center y (normalized 0-1)
    w: float   # Width (normalized 0-1)
    h: float   # Height (normalized 0-1)


class ImageLabels(BaseModel):
    """Labels for a single image."""
    filename: str
    boxes: List[BoundingBox]


class ClassConfig(BaseModel):
    """Class configuration."""
    id: int
    name: str
    color: str


# ============================================================================
# Server State
# ============================================================================

class ServerState:
    """Global server state."""
    
    def __init__(self):
        self.images_dir: Optional[Path] = None
        self.labels_dir: Optional[Path] = None
        self.classes: Dict[int, Dict[str, str]] = DEFAULT_CLASSES.copy()
    
    def setup(self, images_dir: str, labels_dir: Optional[str] = None):
        """Setup directories."""
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        if labels_dir:
            self.labels_dir = Path(labels_dir)
        else:
            # Auto-detect: images/xxx -> labels/xxx
            img_str = str(self.images_dir)
            if 'images' in img_str:
                self.labels_dir = Path(img_str.replace('images', 'labels'))
            else:
                self.labels_dir = self.images_dir / 'labels'
        
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Images: {self.images_dir}")
        print(f"📁 Labels: {self.labels_dir}")
    
    def get_image_list(self) -> List[str]:
        """Get list of image filenames."""
        if not self.images_dir:
            return []
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        images = []
        for f in sorted(self.images_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                images.append(f.name)
        return images
    
    def get_label_path(self, image_filename: str) -> Path:
        """Get label file path for an image."""
        stem = Path(image_filename).stem
        return self.labels_dir / f"{stem}.txt"
    
    def read_labels(self, image_filename: str) -> List[BoundingBox]:
        """Read labels for an image."""
        label_path = self.get_label_path(image_filename)
        if not label_path.exists():
            return []
        
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        boxes.append(BoundingBox(
                            class_id=int(parts[0]),
                            cx=float(parts[1]),
                            cy=float(parts[2]),
                            w=float(parts[3]),
                            h=float(parts[4]),
                        ))
                    except ValueError:
                        continue
        return boxes
    
    def write_labels(self, image_filename: str, boxes: List[BoundingBox]):
        """Write labels for an image in YOLO format."""
        label_path = self.get_label_path(image_filename)
        
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}\n")
    
    def delete_labels(self, image_filename: str):
        """Delete labels for an image."""
        label_path = self.get_label_path(image_filename)
        if label_path.exists():
            label_path.unlink()


# Global state
state = ServerState()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="YOLO Labeling Tool",
    description="Web-based annotation tool for YOLO object detection",
    version="1.0.0"
)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """Get class configuration."""
    return {
        "classes": [
            {"id": k, "name": v["name"], "color": v["color"]}
            for k, v in state.classes.items()
        ],
        "images_dir": str(state.images_dir),
        "labels_dir": str(state.labels_dir),
    }


@app.get("/api/images")
async def list_images() -> Dict[str, Any]:
    """List all images in the eval directory."""
    images = state.get_image_list()
    return {"images": images, "count": len(images)}


@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Serve an image file."""
    # Security: prevent path traversal
    filename = Path(filename).name
    image_path = state.images_dir / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    return FileResponse(image_path)


@app.get("/api/labels/{filename}")
async def get_labels(filename: str) -> ImageLabels:
    """Get labels for an image."""
    filename = Path(filename).name
    boxes = state.read_labels(filename)
    return ImageLabels(filename=filename, boxes=boxes)


@app.post("/api/labels/{filename}")
async def save_labels(filename: str, data: ImageLabels):
    """Save labels for an image."""
    filename = Path(filename).name
    state.write_labels(filename, data.boxes)
    return {"status": "saved", "filename": filename, "count": len(data.boxes)}


@app.delete("/api/labels/{filename}")
async def delete_labels(filename: str):
    """Delete labels for an image."""
    filename = Path(filename).name
    state.delete_labels(filename)
    return {"status": "deleted", "filename": filename}


@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Get labeling statistics."""
    images = state.get_image_list()
    labeled_count = 0
    total_boxes = 0
    
    for img in images:
        boxes = state.read_labels(img)
        if boxes:
            labeled_count += 1
            total_boxes += len(boxes)
    
    return {
        "total_images": len(images),
        "labeled_images": labeled_count,
        "unlabeled_images": len(images) - labeled_count,
        "total_boxes": total_boxes,
        "progress": f"{labeled_count}/{len(images)} ({100*labeled_count/max(len(images),1):.1f}%)",
    }


# ============================================================================
# Static Files & HTML
# ============================================================================

# Mount static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main HTML page."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    else:
        return """
        <html>
            <head><title>YOLO Labeling Tool</title></head>
            <body>
                <h1>YOLO Labeling Tool</h1>
                <p>Static files not found. Please ensure static/index.html exists.</p>
            </body>
        </html>
        """


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO Labeling Tool Server")
    parser.add_argument(
        "--images", "-i",
        default=None,
        help="Path to images directory (uses val from config if not specified)"
    )
    parser.add_argument(
        "--labels", "-l",
        default=None,
        help="Path to labels directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to data.yaml config file (for class names)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="Server port (default: 8765)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Load classes and val path from YAML if provided
    images_path = args.images
    if args.config:
        try:
            classes, val_path = load_classes_from_yaml(args.config)
            state.classes = classes
            print(f"📋 Loaded {len(state.classes)} classes from {args.config}")
            
            # Use val path from yaml if --images not specified
            if not images_path and val_path:
                images_path = val_path
                print(f"📁 Using val path from config: {val_path}")
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}, using defaults")
    
    # Check that we have an images path
    if not images_path:
        print("❌ Error: Must specify --images or provide --config with val path")
        return
    
    # Setup state
    state.setup(images_path, args.labels)
    
    print(f"\n🚀 Starting YOLO Labeling Tool")
    print(f"   Open: http://localhost:{args.port}")
    print(f"   Images: {len(state.get_image_list())} files")
    print(f"   Classes: {[c['name'] for c in state.classes.values()]}\n")
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
