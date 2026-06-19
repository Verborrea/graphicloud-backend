import json
import re

# Mapeos de enums
ICON_CATEGORY = {
    "IconCategory.ARROWS": "arrows",
    "IconCategory.BRAND": "brands",
    "IconCategory.COMMERCE": "commerce",
    "IconCategory.COMMUNICATION": "communications",
    "IconCategory.DESIGN": "design",
    "IconCategory.DEVELOPMENT": "technology & development",
    "IconCategory.EDITOR": "editor",
    "IconCategory.FINANCE": "finances",
    "IconCategory.GAMES": "games",
    "IconCategory.HEALTH": "health & wellness",
    "IconCategory.MAP": "maps & travel",
    "IconCategory.MEDIA": "media",
    "IconCategory.NATURE": "nature",
    "IconCategory.OBJECTS": "objects",
    "IconCategory.OFFICE": "office",
    "IconCategory.PEOPLE": "people",
    "IconCategory.SYSTEM": "system",
    "IconCategory.WEATHER": "weather",
}

FIGMA_CATEGORY = {
    "FigmaCategory.ARROWS": "arrows",
    "FigmaCategory.BRAND": "brands",
    "FigmaCategory.COMMERCE": "commerce",
    "FigmaCategory.COMMUNICATION": "communication",
    "FigmaCategory.DESIGN": "design",
    "FigmaCategory.DEVELOPMENT": "technology & development",
    "FigmaCategory.EDUCATION": "education",
    "FigmaCategory.FINANCE": "math & finance",
    "FigmaCategory.GAMES": "games",
    "FigmaCategory.HEALTH": "health & wellness",
    "FigmaCategory.MAP": "maps & travel",
    "FigmaCategory.MEDIA": "media",
    "FigmaCategory.OFFICE": "office & editing",
    "FigmaCategory.PEOPLE": "people",
    "FigmaCategory.SECURITY": "security & warnings",
    "FigmaCategory.SYSTEM": "system & devices",
    "FigmaCategory.TIME": "time",
    "FigmaCategory.WEATHER": "weather & nature",
}

with open("icons.ts", "r") as f:
    content = f.read()

# Extraer cada bloque { ... } del array
blocks = re.findall(r"\{([^{}]+)\}", content)

result = []

for block in blocks:
    # name
    name_match = re.search(r'name:\s*"([^"]+)"', block)
    if not name_match:
        continue
    name = name_match.group(1)

    # categories -> resuelve los enums
    cats_match = re.search(r"categories:\s*\[([^\]]*)\]", block)
    categories = []
    if cats_match:
        for token in cats_match.group(1).split(","):
            token = token.strip()
            if token in ICON_CATEGORY:
                categories.append(ICON_CATEGORY[token])

    # figma_category -> resuelve el enum
    figma_match = re.search(r"figma_category:\s*(FigmaCategory\.\w+)", block)
    figma_cat = ""
    if figma_match:
        figma_cat = FIGMA_CATEGORY.get(figma_match.group(1), "")

    # tags -> lista de strings, filtra *new*
    tags_match = re.search(r"tags:\s*\[([^\]]*)\]", block, re.DOTALL)
    tags = []
    if tags_match:
        tags = re.findall(r'"([^"]+)"', tags_match.group(1))
        tags = [t for t in tags if t != "*new*"]

    # Armar text: name + categories + figma_category + tags (deduplicado, preservando orden)
    seen = set()
    parts = []
    for word in [name] + categories + ([figma_cat] if figma_cat else []) + tags:
        if word not in seen:
            seen.add(word)
            parts.append(word)

    result.append({"name": name, "text": " ".join(parts)})

with open("icons.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"✅ {len(result)} íconos → icons.json")
