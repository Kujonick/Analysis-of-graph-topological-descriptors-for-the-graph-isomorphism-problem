from pathlib import Path

class Settings():
    dataset_dir             = Path('datasets')
    raw_datasets_dir        = dataset_dir / 'raw'
    raw_metadata_path       = raw_datasets_dir / 'metadata.json'
    processed_datasets_dir  = dataset_dir / 'processed'
    k_graph_dir             = dataset_dir / 'kgraphs'