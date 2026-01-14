"""
Create metadata.json for PVF-10 dataset.
This dataset is already in iron red thermal format.
Supports multiple resolution directories.
"""

import json
from pathlib import Path


def create_pvf10_metadata(dataset_dirs: list, output_path: str):
    """
    Create metadata.json from PVF-10 folder structure.
    
    Args:
        dataset_dirs: List of paths to PVF_10 train directories
        output_path: Output path for metadata.json
    """
    # Class name mapping from folder names
    class_mapping = {
        '01bottom dirt': 'Bottom-Dirt',
        '02break': 'Broken-Cell',
        '03Debris cover': 'Debris-Cover',
        '04junction box heat': 'Junction-Box-Heat',
        '05hot cell': 'Hot-Spot',
        '06shadow': 'Shadow',
        '07short circuit panel': 'Short-Circuit-Panel',
        '08string short circuit': 'String-Short-Circuit',
        '09substring open circuit': 'Substring-Open-Circuit',
        '10healthy panel': 'No-Anomaly'  # Healthy = No-Anomaly
    }
    
    metadata = {}
    image_count = 0
    
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue
            
        resolution = dataset_path.parent.name  # e.g., PVF_10_112x112
        
        print(f"\nScanning: {dataset_path}")
        print(f"Resolution: {resolution}")
        print("-" * 50)
        
        for folder in sorted(dataset_path.iterdir()):
            if not folder.is_dir():
                continue
            
            folder_name = folder.name
            anomaly_class = class_mapping.get(folder_name, folder_name)
            
            # Find all images in folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            images = []
            for ext in image_extensions:
                images.extend(folder.glob(ext))
            
            print(f"  {folder_name}: {len(images)} images -> {anomaly_class}")
            
            for img_path in images:
                img_id = f"pvf10_{image_count:05d}"
                
                # Get relative path from PVF-10 root
                pvf10_root = dataset_path.parent.parent
                
                metadata[img_id] = {
                    'image_filepath': str(img_path.relative_to(pvf10_root)),
                    'anomaly_class': anomaly_class,
                    'source': 'PVF-10',
                    'resolution': resolution,
                    'original_folder': folder_name
                }
                
                image_count += 1
    
    # Save metadata
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 50)
    print(f"Total images: {image_count}")
    print(f"Total classes: {len(set(m['anomaly_class'] for m in metadata.values()))}")
    print(f"Saved: {output_file}")
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create metadata.json for PVF-10 dataset")
    parser.add_argument(
        "--datasets", 
        nargs="+",
        default=[
            r"/mnt/d/ThucTap/Al_platform_Solar/datasets/PVF-10/PVF_10_110x60/train",
            r"/mnt/d/ThucTap/Al_platform_Solar/datasets/PVF-10/PVF_10_112x112/train"
        ],
        help="Paths to PVF-10 train directories"
    )
    parser.add_argument(
        "--output",
        default=r"/mnt/d/ThucTap/Al_platform_Solar/datasets/PVF-10/metadata.json",
        help="Output path for metadata.json"
    )
    
    args = parser.parse_args()
    
    create_pvf10_metadata(args.datasets, args.output)

