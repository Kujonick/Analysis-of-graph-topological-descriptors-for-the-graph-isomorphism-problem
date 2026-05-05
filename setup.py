import requests

from src.settings import Settings
settings = Settings()

datasets = [
    'chordal10',
    'chordal6',
    'chordal7',
    'chordal8',
    'chordal9',
    'eul10',
    'eul6',
    'eul7',
    'eul8',
    'eul9',
    'ge10c',
    'graph4c',
    'graph5',
    'graph6',
    'graph7',
    'graph7c',
    'graph8',
    'graph8c',
    'graph9',
    'graph9c',
    'highlyirregular11',
    'highlyirregular12',
    'highlyirregular13',
    'highlyirregular14',
    'highlyirregular15',
    'perfect6',
    'perfect7',
    'perfect8',
    'perfect9',
    'planar_conn.5',
    'planar_conn.6',
    'planar_conn.7',
    'planar_conn.8',
    'planar_conn.9',
    'selfcomp12',
    'selfcomp13',
    'selfcomp9',
    'sr291467',
    'sr351668',
    'sr351899',
    'sr361446',
    'sr361566',
    'sr371889some',
    'sr401224',
    'sr65321516some'
]

def download_datasets(names):
    base_url = "https://users.cecs.anu.edu.au/~bdm/data"
    settings.ensure_paths()

    for name in names:
        url = f"{base_url}/{name}.g6"
        out_path = Settings.raw_datasets_dir / f"{name}.g6"

        if out_path.exists():
            continue

        print(f"Downloading {name}...")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        out_path.write_bytes(response.content)

        print(f"Saved to {out_path}")

if __name__ == "__main__":
    download_datasets(datasets)