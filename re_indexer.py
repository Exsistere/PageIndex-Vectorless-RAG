import json
import os
with open('sample_treeindex.json', 'r') as file:
    data = json.load(file)

for node in data["roots"]:
    node["start_page"] += 25
    node["end_page"] += 25
    if "children" in node:
        for child_node in node["children"]:
            child_node["start_page"] += 25
            child_node["end_page"] += 25

with open('temp.json', 'w') as file:
    json.dump(data, file, indent=4)

os.replace('temp.json', 'sample_treeindex.json')