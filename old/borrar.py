import json

file1 = "icons.json"
file2 = "icons_enriched_2.json"
output_file = "merged.json"


def merge_texts(text1: str, text2: str) -> str:
    words = []
    seen = set()

    for word in (text1 + " " + text2).split():
        if word not in seen:
            seen.add(word)
            words.append(word)

    return " ".join(words)


with open(file1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(file2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# Indexar por name
data2_by_name = {item["name"]: item for item in data2}

merged = []

for item1 in data1:
    name = item1["name"]

    if name in data2_by_name:
        item2 = data2_by_name[name]

        merged.append(
            {
                "name": name,
                "text": merge_texts(item1.get("text", ""), item2.get("text", "")),
            }
        )
    else:
        merged.append(item1)

# Si hubiera nombres extra en data2 que no están en data1
existing_names = {item["name"] for item in merged}

for item2 in data2:
    if item2["name"] not in existing_names:
        merged.append(
            {"name": item2["name"], "text": merge_texts(item2.get("text", ""), "")}
        )

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Guardado en {output_file}")
