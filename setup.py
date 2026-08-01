import requests
import gzip
from src.settings import Settings
import shutil
settings = Settings()
gzipped_files = {
    "sr361566",
}
base_url_cecs = "https://users.cecs.anu.edu.au/~bdm/data"
datasets_cecs = [
    "chordal10",
    "chordal6",
    "chordal7",
    "chordal8",
    "chordal9",
    "eul10",
    "eul6",
    "eul7",
    "eul8",
    "eul9",
    "ge10c",
    "graph4c",
    "graph5",
    "graph6",
    "graph7",
    "graph7c",
    "graph8",
    "graph8c",
    "graph9",
    "graph9c",
    "highlyirregular11",
    "highlyirregular12",
    "highlyirregular13",
    "highlyirregular14",
    "highlyirregular15",
    "perfect6",
    "perfect7",
    "perfect8",
    "perfect9",
    "planar_conn.5",
    "planar_conn.6",
    "planar_conn.7",
    "planar_conn.8",
    "planar_conn.9",
    "selfcomp12",
    "selfcomp13",
    "selfcomp9",
    "sr291467",
    "sr351668",
    "sr351899",
    "sr361446",
    "sr361566",
    "sr371889some",
    "sr401224",
    "sr65321516some",
]
base_url_brec = (
    "https://github.com/GraphPKU/BREC/raw/refs/heads/Release/customize/Data/raw"
)
datasets_brec = ["4vtx", "basic", "cfi", "dr", "extension", "regular", "str"]


def download_datasets():

    params = [
        (base_url_cecs, datasets_cecs, "g6", Settings.raw_datasets_dir),
        (base_url_brec, datasets_brec, "npy", Settings.brec_byproduct_graphs),
    ]
    settings.ensure_paths()

    for base_url, names, format, directory in params:
        for name in names:
            url = f"{base_url}/{name}.{format}"
            out_path = directory / f"{name}.{format}"

            if out_path.exists():
                continue

            print(f"Downloading {name}...")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            gzipped = name in gzipped_files
            if gzipped:
                out_path = directory / f"{name}.{format}.gz"

            out_path.write_bytes(response.content)

            if gzipped:
                unzipped_file = directory / f"{name}.{format}"
                with gzip.open(out_path, "rb") as f_in:
                    with open(unzipped_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                out_path.unlink()
            print(f"Saved to {out_path}")


if __name__ == "__main__":
    download_datasets()
