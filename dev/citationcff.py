import yaml
from pathlib import Path
from datetime import date

citation_path = Path(__file__).parents[1].joinpath('CITATION.cff')
cita = yaml.safe_load(open(citation_path).read())
cita['date-released'] = date.today()
yaml.safe_dump(cita, open(citation_path, 'w'))
