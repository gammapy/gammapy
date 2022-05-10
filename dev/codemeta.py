"""Utility script to work with the codemeta.json file."""
""" A preliminary file is created with https://codemeta.github.io/codemeta-generator/ without the authors list. Then, one add the authors list stored in CITATION.cff."""

import json
import yaml

FILENAME = "../codemeta.json"

# Update the Author list
CITATIONFILE = "../CITATION.cff"
with open(CITATIONFILE, "r") as f:
    citation = yaml.safe_load(f)
for author in citation["authors"]:
   author["@type"] = "Person"
   author["givenName"] = author.pop("given-names")
   author["familyName"] = author.pop("family-names")
   if "orcid" in author:
       author["@id"] = author.pop("orcid")
   if "affiliation" in author:
       author["affiliation"] = {"@type": "Organization", "name": author["affiliation"]}

# add potentially missing content
with open(FILENAME, "r") as f:
    data = json.load(f)
data["maintainer"] = data["author"][0]
data["readme"] = "https://gammapy.org"
data["issueTracker"] = "https://github.com/gammapy/gammapy/issues"
data["author"] = citation["authors"]
data["identifier"] = citation["identifiers"]

with open(FILENAME, "w") as f:
    json.dump(data, f, indent=4)

# replace bad labelled attributes
with open(FILENAME, "r") as f:
    content = f.read()
content = content.replace("legalName", "name")
content = content.replace("version", "softwareVersion")

with open(FILENAME, "w") as f:
    f.write(content)
