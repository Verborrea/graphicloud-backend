import json

import requests

# Colecciones permitidas
COLLECTIONS = [
    "streamline",
    "tabler",
    "mdi",
    "solar",
    "lucide",
    "heroicons",
    "carbon",
    "fluent",
    "material-symbols",
    "openmoji",
]

BASE_URL = "https://api.iconify.design/collection?prefix="


def extract_icons(data: dict):
    icons = set()

    if "categories" in data and isinstance(data["categories"], dict):
        for icon_list in data["categories"].values():
            icons.update(icon_list)

    if "uncategorized" in data and isinstance(data["uncategorized"], list):
        icons.update(data["uncategorized"])

    if "icons" in data and isinstance(data["icons"], dict):
        icons.update(data["icons"].keys())

    return sorted(icons)


def download_collection(prefix: str):
    url = BASE_URL + prefix
    print(f"Descargando {prefix} …")

    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error descargando {prefix}: {r.status_code}")
        return []

    data = r.json()
    return extract_icons(data)


def main():
    icon_index = {}

    for prefix in COLLECTIONS:
        if prefix in icon_index and len(icon_index[prefix]) > 0:
            print(f"{prefix} ya existe.")
            continue

        icons = download_collection(prefix)

        if icons is None:
            print(f"No se pudo obtener la colección {prefix}")
            continue

        icon_index[prefix] = icons
        print(f"{prefix}: {len(icons)} íconos añadidos")

    # Guardar archivo final
    with open("icons_index.json", "w", encoding="utf-8") as f:
        json.dump(icon_index, f, ensure_ascii=False, indent=2)

    print("\nArchivo generado: icons_index.json")


if __name__ == "__main__":
    main()
