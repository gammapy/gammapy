.. include:: ../references.txt

.. _gammapy_{{ versiondata.version | replace('.', 'p') }}_release:

{% set parts = versiondata.date.split('-') %}
{% set year = parts[0] %}
{% set month = parts[1]|int %}
{% set day = parts[2]|int %}
{% set month_name = ["January","February","March","April","May","June","July","August","September","October","November","December"][month-1] %}

{% set suffix = "th" %}
{% if day in [1,21,31] %}
  {% set suffix = "st" %}
{% elif day in [2,22] %}
  {% set suffix = "nd" %}
{% elif day in [3,23] %}
  {% set suffix = "rd" %}
{% endif %}


{% if render_title %}
{% if versiondata.name %}
{% set title_text = versiondata.version ~ " (" ~ day|string ~ suffix ~ " " ~ month_name ~ ", " ~ year ~ ")" %}
{{ title_text }}
{{ top_underline * title_text|length }}
{% endif %}
{% endif %}

- Released {{ day }}{{ suffix }} {{ month_name }}, {{ year }}
- contributors
- pull requests since v (not all listed below)
- closed issues

Summary
-------

{% for category, val in definitions.items() %}
{% set category_name = definitions[category]['name'] %}
{% set show_content = definitions[category]['showcontent'] %}
{% set underline = underlines[0] %}

{{ category_name }}
{{ underline * category_name|length }}

{% set underline = underlines[1] %}
{% for section, section_categories in sections.items() %}
{% if section and category in section_categories %}
{{ section }}
{{ underline * section|length }}
{% endif %}

{% if section_categories and category in section_categories %}
{% if show_content %}
{% for text, values in section_categories[category].items() %}
- {{ text }} [{{ values|join(', ') }}]
{% endfor %}
{% else %}
- {{ section_categories[category]['']|join(', ') }}
{% endif %}

{% if section_categories[category]|length == 0 %}
No significant changes.
{% endif %}
{% endif %}
{% endfor %}
{% endfor %}

