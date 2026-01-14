"""
OBB Detector Module
===================
Handles YOLO OBB detection for solar panels.
"""

from ultralytics import YOLO


class OBBDetector:
    """YOLO OBB detector for solar panels."""
    
    def __init__(self, weights_path: str):
        """
        Initialize detector.
        
        Args:
            weights_path: Path to YOLO OBB model weights
        """
        print(f"Loading YOLO OBB model from: {weights_path}")
        self.model = YOLO(weights_path)
        self.class_names = self.model.names  # Class index to name mapping
    
    def detect(self, image_path: str, conf_threshold: float = 0.5, 
               filter_class: str = None) -> list:
        """
        Run YOLO OBB detection on image.
        
        Args:
            image_path: Path to image
            conf_threshold: Minimum confidence to accept detection
            filter_class: Only return detections of this class (e.g., "No-Anomaly")
            
        Returns:
            List of OBB detections: [{cx, cy, width, height, angle, confidence, class_name}, ...]
        """
        results = self.model(image_path, verbose=False)
        
        # First pass: collect all valid detections with areas
        raw_detections = []
        for result in results:
            if result.obb is not None:
                obbs = result.obb.xywhr.cpu().numpy()
                confs = result.obb.conf.cpu().numpy()
                classes = result.obb.cls.cpu().numpy()  # Class indices
                
                for obb, conf, cls_idx in zip(obbs, confs, classes):
                    if conf < conf_threshold:
                        continue
                    
                    # Get class name
                    class_name = self.class_names.get(int(cls_idx), "Unknown")
                    
                    # Filter by class if specified
                    if filter_class is not None and class_name != filter_class:
                        continue
                    
                    cx, cy, w, h, angle = obb
                    if angle > 25 and angle < 75:
                        continue
                    
                    area = w * h
                    raw_detections.append({
                        'cx': float(cx),
                        'cy': float(cy),
                        'width': float(w),
                        'height': float(h),
                        'angle': float(angle),
                        'confidence': float(conf),
                        'class_name': class_name,
                        'area': float(area)
                    })
        
        # Calculate average area
        if not raw_detections:
            return []
        
        avg_area = sum(d['area'] for d in raw_detections) / len(raw_detections)
        
        # Second pass: filter by area > average
        detections = [d for d in raw_detections if d['area'] >= avg_area]
        
        return detections

