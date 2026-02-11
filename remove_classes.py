import xml.etree.ElementTree as ET
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Classes to remove
CLASSES_TO_REMOVE = {
    'train',
    'trailer', 
    'caravan',
    'covered_vehicle',
    'overloaded_vehicle',
    'ego vehicle'
}

def process_xml_file(xml_path):
    """Remove specific classes from an XML annotation file"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        removed_count = 0
        
        # Remove objects with specified class names
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None and name_elem.text in CLASSES_TO_REMOVE:
                root.remove(obj)
                removed_count += 1
        
        # Only save if we removed something
        if removed_count > 0:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        return removed_count
        
    except Exception as e:
        return -1

def main():
    annotations_dir = Path('/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/datasets/Annotations')
    
    # Get all XML files
    xml_files = list(annotations_dir.glob('*.xml'))
    print(f"Found {len(xml_files)} annotation files")
    print(f"Removing classes: {', '.join(sorted(CLASSES_TO_REMOVE))}\n")
    
    # Process files with multiprocessing
    num_workers = min(cpu_count(), 16)
    print(f"Processing with {num_workers} workers...\n")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_xml_file, xml_files),
            total=len(xml_files),
            desc="Removing classes"
        ))
    
    # Aggregate results
    total_removed = sum(r for r in results if r > 0)
    files_modified = sum(1 for r in results if r > 0)
    errors = sum(1 for r in results if r < 0)
    
    # Print summary
    print("\n" + "="*60)
    print("REMOVAL SUMMARY")
    print("="*60)
    print(f"Files processed:  {len(xml_files)}")
    print(f"Files modified:   {files_modified}")
    print(f"Objects removed:  {total_removed}")
    if errors > 0:
        print(f"Errors:           {errors}")
    print("="*60)

if __name__ == '__main__':
    main()
