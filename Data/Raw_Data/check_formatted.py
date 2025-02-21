import json

with open("formatted_testdata.json", "r", encoding="utf-8") as f:
    formatted_data = json.load(f)

print("Formatted dataset preview (first 2 entries):")
for entry in formatted_data[:2]:
    print(entry)
    print("---")