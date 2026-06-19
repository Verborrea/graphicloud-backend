import json
import time

from openai import OpenAI

client = OpenAI(api_key="...")

with open("icons.json") as f:
    icons = json.load(f)

BATCH_SIZE = 50

SYSTEM_PROMPT = """You receive a list of icons with name and existing keywords.
For each icon, expand the 'text' field with more relevant keywords: 
synonyms, use cases, related concepts, UI contexts where this icon would appear.
Keep it concise (under 20 words total), no repetition, all lowercase, no punctuation.
Return ONLY a JSON array in the same order, each object with "name" and "text"."""


def enrich_batch(batch):
    payload = json.dumps([{"name": i["name"], "text": i["text"]} for i in batch])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    if isinstance(data, list):
        return data
    return next(v for v in data.values() if isinstance(v, list))


results = []
total_batches = (len(icons) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(icons), BATCH_SIZE):
    batch = icons[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    print(f"{batch_num}/{total_batches} ({len(batch)} íconos)...")

    try:
        enriched = enrich_batch(batch)
        results.extend(enriched)
        time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
        results.extend(batch)

with open("icons_enriched_2.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(len(results))
