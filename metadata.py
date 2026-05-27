import json
import re

with open("icons.ts", "r") as f:
    content = f.read()

TARGET_ICONS = [
    "brain",
    "graph",
    "chart-bar",
    "chart-line",
    "file-text",
    "magnifying-glass",
    "books",
    "cpu",
    "arrows-clockwise",
    "chart-scatter",
]

pattern = re.compile(r'\{\s*\n\s+name:\s*"([^"]+)".*?\n\s*\}', re.DOTALL)
results = []

for m in pattern.finditer(content):
    name = m.group(1)
    if name not in TARGET_ICONS:
        continue
    block = m.group(0)

    cats = re.findall(r"IconCategory\.(\w+)", block)
    cats = [c.lower().replace("_", " ") for c in cats]

    tags_raw = re.findall(r'"([^"]+)"', block)
    # Filtrar el nombre mismo y los marcadores *new* *updated*
    tags = [t for t in tags_raw if t != name and not re.match(r"\*\w+\*", t)]

    enriched = name.replace("-", " ") + " " + " ".join(cats) + " " + " ".join(tags)
    enriched = enriched.strip()

    results.append({"name": name, "categories": cats, "tags": tags, "text": enriched})

results.sort(key=lambda x: TARGET_ICONS.index(x["name"]))
print(json.dumps(results, indent=2, ensure_ascii=False))
