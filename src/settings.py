from pathlib import Path

class Settings():
    dataset_dir             = Path('datasets')
    raw_datasets_dir        = dataset_dir / Path('raw')
    raw_metadata_path       = raw_datasets_dir / Path('metadata.json')
    processed_datasets_dir  = dataset_dir / Path('processed')