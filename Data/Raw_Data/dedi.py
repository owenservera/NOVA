import json
from collections import defaultdict

with open("testdata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Remove duplicates (simple hash-based deduplication)
seen = set()
cleaned_data = []
for entry in data:
    entry_str = str(entry)
    if entry_str not in seen:
        seen.add(entry_str)
        cleaned_data.append(entry)

with open("cleaned_testdata.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)

print(f"Removed duplicates. New size: {len(cleaned_data)} entries")
