from pathlib import Path


class Settings:
    dataset_dir = Path("datasets")
    raw_datasets_dir = dataset_dir / "raw"
    raw_metadata_path = raw_datasets_dir / "metadata.json"
    processed_datasets_dir = dataset_dir / "processed"
    k_graph_dir = dataset_dir / "kgraphs"
    brec_byproduct_graphs = dataset_dir / "brec/Data/raw"

    @classmethod
    def ensure_paths(cls):
        dirs = [
            cls.dataset_dir,
            cls.raw_datasets_dir,
            cls.processed_datasets_dir,
            cls.k_graph_dir,
            cls.raw_metadata_path.parent,
            cls.brec_byproduct_graphs,
        ]

        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
