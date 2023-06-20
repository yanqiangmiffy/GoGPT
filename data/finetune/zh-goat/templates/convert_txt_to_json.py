import json

# Open the instruction template text file
with open('template.txt', 'r', encoding='utf-8') as f_in:

    lines = {}

    for i, line in enumerate(f_in):
        lines[str(i+1)] = line.strip()

# Write the list to a JSON file
with open('goat.json', 'w', encoding='utf-8') as f_out:
    json.dump(lines, f_out, indent=4,ensure_ascii=False)
