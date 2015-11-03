# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from flask_wtf import Form
from wtforms import SelectField, StringField
from wtforms.validators import InputRequired

__all__ = ['CatalogBrowserForm']


catalog_name_choices = [
    ('3fgl', '3fgl'),
    ('2fhl', '2fhl'),
    ('hgps', 'hgps'),
]

info_display_choices = [
    ('Table', 'Table'),
    ('Spectrum', 'Spectrum'),
    ('Image', 'Image'),
]


class CatalogBrowserForm(Form):
    catalog_name = SelectField(
        label='Catalog Name',
        default='3fgl',
        choices=catalog_name_choices,
        validators=[InputRequired()]
    )
    source_name = StringField(
        label='Source Name',
        default='3FGL J0349.9-2102',
        validators=[InputRequired()]
    )
    info_display = SelectField(
        label='Show Info',
        default='Table',
        choices=info_display_choices,
        validators=[InputRequired()]
    )
