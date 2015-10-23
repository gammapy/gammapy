from flask_wtf import Form
from wtforms import SelectField, StringField
from wtforms.validators import InputRequired

__all__ = ['CatalogBrowserForm']


catalog_name_choices = [
    ('3FGL', '3FGL'),
    ('2FHL', '2FHL'),
]

info_display_choices = [
    ('Table', 'Table'),
    ('Spectrum', 'Spectrum'),
    ('Image', 'Image'),
]


class CatalogBrowserForm(Form):
    catalog_name = SelectField(
        label='Catalog Name',
        default='3FGL',
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
