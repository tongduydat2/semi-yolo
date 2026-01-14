"""
Two-Stage Detector
==================
Combines OBB detection (localization) + Classification (defect type).

Pipeline:
1. OBB detector finds all panel/cell regions
2. Each detection is cropped and classified
3. Final output: detections with class labels from classifier

Usage:
    python two_stage_detector.py --image input.jpg --visualize
    python two_stage_detector.py --source folder/ --output results/
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


class TwoStageDetector:
    """Two-stage detector combining OBB localization and classification."""
    
    # Mapping from PVF-10 folder names to standard class names
    CLASS_MAPPING = {
        '01bottom dirt': 'Bottom-Dirt',
        '02break': 'Broken-Cell',
        '03Debris cover': 'Debris-Cover',
        '04junction box heat': 'Junction-Box-Heat',
        '05hot cell': 'Hot-Spot',
        '06shadow': 'Shadow',
        '07short circuit panel': 'Short-Circuit-Panel',
        '08string short circuit': 'String-Short-Circuit',
        '09substring open circuit': 'Substring-Open-Circuit',
        '10healthy panel': 'No-Anomaly',
    }
    
    def __init__(self, obb_weights: str, classifier_weights: str):
        """
        Initialize two-stage detector.
        
        Args:
            obb_weights: Path to YOLO OBB model weights
            classifier_weights: Path to YOLO classifier weights
        """
        print(f"Loading OBB detector: {obb_weights}")
        self.obb_model = YOLO(obb_weights)
        
        print(f"Loading Classifier: {classifier_weights}")
        self.classifier = YOLO(classifier_weights)
        
        # Get classifier class names and create reverse mapping
        self.classifier_names = self.classifier.names
        self.class_name_map = self._build_class_map()
        
        print(f"OBB classes: {self.obb_model.names}")
        print(f"Classifier classes: {self.classifier_names}")
    
    def _build_class_map(self) -> dict:
        """Build mapping from classifier index to standard class name."""
        mapping = {}
        for idx, name in self.classifier_names.items():
            # Try to match with CLASS_MAPPING
            if name in self.CLASS_MAPPING:
                mapping[idx] = self.CLASS_MAPPING[name]
            else:
                # Use name directly if no mapping found
                mapping[idx] = name
        return mapping
    
    def crop_obb(self, image: np.ndarray, obb: dict, padding: float = 0.1) -> np.ndarray:
        """
        Crop image region based on OBB (oriented bounding box).
        
        Args:
            image: Input image (BGR)
            obb: OBB dict with cx, cy, width, height, angle
            padding: Padding ratio around OBB
            
        Returns:
            Cropped and rotated image region
        """
        cx, cy = obb['cx'], obb['cy']
        w, h = obb['width'], obb['height']
        angle = obb['angle']  # In radians
        
        # Add padding
        w_padded = w * (1 + padding)
        h_padded = h * (1 + padding)
        
        # Get rotation matrix
        angle_deg = np.degrees(angle)
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        
        # Rotate entire image
        img_h, img_w = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (img_w, img_h))
        
        # Crop the axis-aligned bounding box
        x1 = max(0, int(cx - w_padded / 2))
        y1 = max(0, int(cy - h_padded / 2))
        x2 = min(img_w, int(cx + w_padded / 2))
        y2 = min(img_h, int(cy + h_padded / 2))
        
        crop = rotated[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
            
        return crop
    
    def classify_crop(self, crop: np.ndarray, conf_threshold: float = 0.3) -> dict:
        """
        Classify a cropped panel/cell image.
        
        Args:
            crop: Cropped image (BGR)
            conf_threshold: Minimum confidence
            
        Returns:
            Dict with class_name and confidence
        """
        # Run classification
        results = self.classifier(crop, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            probs = result.probs
            
            if probs is not None:
                top1_idx = probs.top1
                top1_conf = float(probs.top1conf)
                
                if top1_conf >= conf_threshold:
                    class_name = self.class_name_map.get(top1_idx, 
                                                         self.classifier_names.get(top1_idx, "Unknown"))
                    return {
                        'class_name': class_name,
                        'confidence': top1_conf,
                        'class_idx': top1_idx
                    }
        
        return {'class_name': 'Unknown', 'confidence': 0.0, 'class_idx': -1}
    
    def detect(self, image_path: str, 
               obb_conf: float = 0.5, 
               cls_conf: float = 0.3) -> list:
        """
        Run two-stage detection on an image.
        
        Args:
            image_path: Path to input image
            obb_conf: OBB detection confidence threshold
            cls_conf: Classification confidence threshold
            
        Returns:
            List of detections with OBB coordinates and class labels
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Stage 1: OBB detection
        obb_results = self.obb_model(image_path, verbose=False)
        
        detections = []
        for result in obb_results:
            if result.obb is None:
                continue
                
            obbs = result.obb.xywhr.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            obb_classes = result.obb.cls.cpu().numpy()
            
            for obb_data, conf, obb_cls in zip(obbs, confs, obb_classes):
                if conf < obb_conf:
                    continue
                
                cx, cy, w, h, angle = obb_data
                obb = {
                    'cx': float(cx),
                    'cy': float(cy),
                    'width': float(w),
                    'height': float(h),
                    'angle': float(angle),
                    'obb_confidence': float(conf),
                    'obb_class': self.obb_model.names.get(int(obb_cls), "Unknown")
                }
                
                # Stage 2: Crop and classify
                crop = self.crop_obb(image, obb)
                
                if crop is not None and crop.size > 0:
                    cls_result = self.classify_crop(crop, cls_conf)
                    if cls_result['class_name'] == "No-Anomaly":
                        continue
                    obb['class_name'] = cls_result['class_name']
                    obb['class_confidence'] = cls_result['confidence']
                else:
                    obb['class_name'] = 'Unknown'
                    obb['class_confidence'] = 0.0
                
                detections.append(obb)
        
        return detections
    
    def visualize(self, image_path: str, detections: list, 
                  output_path: str = None) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image_path: Path to input image
            detections: List of detections from detect()
            output_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        image = cv2.imread(str(image_path))
        
        # Color map for classes
        colors = {
            'No-Anomaly': (0, 255, 0),      # Green
            'Hot-Spot': (0, 0, 255),         # Red
            'Broken-Cell': (0, 165, 255),    # Orange
            'Bottom-Dirt': (255, 0, 255),    # Magenta
            'Shadow': (128, 128, 128),       # Gray
            'Debris-Cover': (255, 255, 0),   # Cyan
            'Junction-Box-Heat': (0, 255, 255),  # Yellow
            'Short-Circuit-Panel': (255, 0, 0),  # Blue
            'String-Short-Circuit': (128, 0, 128),  # Purple
            'Substring-Open-Circuit': (0, 128, 255),  # Orange-red
            'Unknown': (200, 200, 200),      # Light gray
        }
        
        for det in detections:
            cx, cy = det['cx'], det['cy']
            w, h = det['width'], det['height']
            angle = det['angle']
            class_name = det['class_name']
            conf = det.get('class_confidence', det.get('obb_confidence', 0))
            
            if class_name == 'Unknown':
                continue
            # Get OBB corner points
            rect = ((cx, cy), (w, h), np.degrees(angle))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get color
            color = colors.get(class_name, (200, 200, 200))
            
            # Draw OBB
            cv2.drawContours(image, [box], 0, color, 1)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_pos = (int(cx - w/2), int(cy - h/2 - 10))
            cv2.putText(image, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"Saved visualization to: {output_path}")
        
        return image


def is_video_file(path: Path) -> bool:
    """Check if file is a video based on extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    return path.suffix.lower() in video_extensions


def process_video(detector, video_path: str, output_dir: Path, 
                  obb_conf: float, cls_conf: float, visualize: bool):
    """
    Process video file frame by frame.
    
    Args:
        detector: TwoStageDetector instance
        video_path: Path to video file
        output_dir: Output directory
        obb_conf: OBB confidence threshold
        cls_conf: Classification confidence threshold
        visualize: Whether to save visualization video
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return {}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Setup video writer if visualize
    video_writer = None
    if visualize:
        video_name = Path(video_path).stem
        output_video = output_dir / f"vis_{video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    all_results = {}
    frame_idx = 0
    
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    # Create frames directory if visualize
    frames_dir = None
    if visualize:
        frames_dir = output_dir / f"frames_{Path(video_path).stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame temporarily for detection
        temp_frame_path = output_dir / f"_temp_frame.jpg"
        cv2.imwrite(str(temp_frame_path), frame)
        
        # Detect on frame
        try:
            detections = detector.detect(str(temp_frame_path), obb_conf, cls_conf)
        except:
            detections = []
        
        # Store results
        all_results[f"frame_{frame_idx:06d}"] = detections
        
        # Visualize
        if visualize:
            vis_frame = detector.visualize(str(temp_frame_path), detections)
            
            # Save to video
            if video_writer:
                video_writer.write(vis_frame)
            
            # Save individual frame image
            if frames_dir:
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), vis_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    if video_writer:
        video_writer.release()
        print(f"Saved video to: {output_dir / f'vis_{Path(video_path).stem}.mp4'}")
    
    if frames_dir:
        print(f"Saved {frame_idx} frames to: {frames_dir}")
    
    # Cleanup temp file
    if temp_frame_path.exists():
        temp_frame_path.unlink()
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Two-stage OBB + Classification detector')
    parser.add_argument('--obb-weights', type=str, required=True,
                        help='Path to OBB detector weights')
    parser.add_argument('--cls-weights', type=str, required=True,
                        help='Path to classifier weights')
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--source', type=str, help='Directory or video file to process')
    parser.add_argument('--output', type=str, default='runs/two_stage',
                        help='Output directory')
    parser.add_argument('--obb-conf', type=float, default=0.5,
                        help='OBB detection confidence threshold')
    parser.add_argument('--cls-conf', type=float, default=0.3,
                        help='Classification confidence threshold')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images/video')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = TwoStageDetector(args.obb_weights, args.cls_weights)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Check if source is video
    if args.source and is_video_file(Path(args.source)):
        print(f"\nProcessing video: {args.source}")
        all_results = process_video(
            detector, args.source, output_dir,
            args.obb_conf, args.cls_conf, args.visualize
        )
    else:
        # Get image list
        if args.image:
            images = [Path(args.image)]
        elif args.source:
            source_dir = Path(args.source)
            if source_dir.is_file():
                images = [source_dir]
            else:
                images = list(source_dir.glob('*.jpg')) + \
                         list(source_dir.glob('*.png')) + \
                         list(source_dir.glob('*.JPG'))
        else:
            print("Error: Specify --image or --source")
            return
        
        print(f"\nProcessing {len(images)} images...")
        
        for img_path in images:
            print(f"\nProcessing: {img_path.name}")
            
            # Detect
            detections = detector.detect(str(img_path), args.obb_conf, args.cls_conf)
            
            print(f"  Found {len(detections)} detections:")
            for det in detections:
                print(f"    - {det['class_name']}: {det.get('class_confidence', 0):.2f}")
            
            # Save results
            all_results[img_path.name] = detections
            
            # Visualize
            if args.visualize:
                vis_path = output_dir / f"vis_{img_path.name}"
                detector.visualize(str(img_path), detections, str(vis_path))
    
    # Save JSON results
    json_path = output_dir / "detections.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {json_path}")


if __name__ == '__main__':
    main()
