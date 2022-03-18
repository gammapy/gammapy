"""Utility script to work with the codemeta.json file"""
import json

FILENAME = "../codemeta.json"

# add posstially missing content
with open(FILENAME, "r") as f:
    data = json.load(f)
data["maintainer"] = data["author"][0]
data["readme"] = "https://gammapy.org"
data["issuetracker"] = "https://github.com/gammapy/gammapy/issues"

with open(FILENAME, "w") as f:
    json.dump(data, f, indent=4)

# replace content
with open(FILENAME, "r") as f:
    content = f.read()
content = content.replace("legalName", "name")
content = content.replace("version", "softwareVersion")

with open(FILENAME, "w") as f:
    f.write(content)
