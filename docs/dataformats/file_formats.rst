.. include:: ../references.txt

.. _dataformats_file_formats:

File formats
============

.. note:: This section is not very useful for astronomers trying to get some analysis done.
    If this is you, maybe try to use the search field to find the specific info / method you want?
    The info is for developers or advanced users that are writing analysis scripts.

This section gives an introdution to the various file formats used in various
parts of Gammapy (and gamma-ray astronomy in general) as well as pointers
how to work with data in these formats and learn more about them.

It also contains comments on the strengths and weaknesses of each format,
which are certainly subjective, but might be useful to help you pick a
certain format if you ever want to store or exchange some data
where no suitable standardised format exists.

Introduction
------------

In Gammapy we use existing file formats by `Fermi-LAT`_ and `CTA`_ where available.

This increases inter-operability with the `Fermi ScienceTools`_
and `ctools`_ as well as mind share with users of those tools.

We also introduce new file formats for things that should be easy to store and exchange,
but no well-defined format exists. E.g. we could define a Gaussian PSF in XML format as

.. code-block:: xml

   <psf type="gauss">
       <parameter name="amplitude" value="4.2"/>
       <parameter name="stddev"    value="0.073"/>
   </psf>

Or we could define a JSON file format for fit results:

.. code-block:: json

   {
       "convergence": true,
       "sources": [
           {
               "type": "point",
               "parameters": { "y": 3.2, "x": 4.9, "flux": 99 }
           },
           {
               "type": "gauss",
               "parameters": { "y": -2.3, "x": 3.3, "stddev": 0.13, "flux": 49 }
           }
       ],
       "likelihood": 4.2
   }

By using general-purpose, flexible file formats (XML and JSON in the examples above)
we can store and exchange any information between tools written in any programming language
(that has an XML or JSON library for parsing and generating data in that format).
All we have to do it agree on the structure
(e.g. to use XML and the fact that there's ``psf`` and ``parameter`` elements,
and that ``parameter`` elements have ``name`` and ``value`` attributes)
and semantics (e.g. that the ``stddev`` parameter of the ``gauss`` PSF is the Gaussian width in degrees).

If we don't write the structure down somewhere everyone will invent their own format,
severly limiting our ability as a community to share results and scripts and build up analysis pipelines
without having to write data converter scripts all the time.
To illustrate this issue, note that the PSF information given above could just as well have been
stored in this incompatible format:

.. code-block:: xml

   <gauss_psf>
       <norm>4.2<norm/>
       <sigma>0.073<sigma/>
   </psf>


Note that this is the best we can do in Gammapy at this time where no
final data format specifications for CTA exist.
We hope that some of these formats will be considered useful prototypes for CTA and adopted.
We do not give any guarantees that the formats described here will be supported in the future!
In most cases the CTA collaboration will probably specify other formats
and we'll update Gammapy to use those.

The data format specifications at http://dataprotocols.org/ are a good example
how to specify formats and schemas in an easy-to-understand way.
After all most people that develop gamma-ray analysis software and have to
work with those data files and codes are astronomers with little computer science background.

Overview
--------

The following table gives an overview of the file formats that you'll probably
encounter at some point in your life as a gamma-ray astronomer.

====== ========= ==== ===== ===== ===== ======
Format File type Supported data content Schema
------ --------- ---------------------- ------
\                Meta Table Array Tree
====== ========= ==== ===== ===== ===== ======
INI    text      Yes  No    No    No    Yes
CSV    text      No   Yes   No    No    Yes
JSON   text      Yes  Yes   Yes   Yes   Yes
XML    text      Yes  Yes   Yes   Yes   Yes
FITS   binary    Some Yes   Yes   No    No
ROOT   binary    No   Yes   Yes   Yes   No
====== ========= ==== ===== ===== ===== ======


Almost all entries in the above table are debatable ... here's some caveats:

* The definition of "text" or "binary" file type given here should be read as
  "are files of this type in gamma-ray astronomy commonly opened up in text editors"?
  In reality the distinction is not always clear, e.g. XML can contain binary data
  and FITS contains text headers.
* The "supported data content" should be read as "is commonly used for this kind of content".
  E.g. I put FITS as "no" for tree data (a.k.a. structured or hierarchical data such as
  in the JSON example above) even though people have found ways to encode such information
  in FITS headers or data extensions.
* The schema supports is best (very common, well-understood, good tools) for `XML schema`_,
  but there's some schema support for the other formats as well.
  This will be discussed in the section `Validation`_ below.

Here's a short description of each format with references if you want to learn more:

INI
+++

**INI** files (see `Wikipedia <https://en.wikipedia.org/wiki/INI_file>`__)
are the most easy to write and edit for humans and can contain ``#`` comments
and are thus a good for configuration files.
file extensions of ``.ini``, ``.conf`` and ``.cfg`` are common.
Astropy bundles `configobj <http://configobj.readthedocs.io/>`__ to read, write and validate
INI files ... to use it in your code

.. code-block:: python

   from astropy.extern.configobj import configobj, validate

Unfortunately INI files are not standardised, so there's only conventions and tons of variants.

.. _CSV_files:

CSV
+++

**CSV** files (see `Wikipedia <https://en.wikipedia.org/wiki/Comma-separated_values>`__),
store tables as comma-separated values (or tab or whitespace separated),
sometimes with the column names in the first row, sometimes with ``#`` comments.
The good thing is that you can import and export data in CSV format from all spreadsheet
programs (e.g. `Microsoft Excel <https://en.wikipedia.org/wiki/Microsoft_Excel>`__,
`Apple Keynote <https://en.wikipedia.org/wiki/Keynote_(presentation_software)>`__ or
`LibreOffice Calc <https://en.wikipedia.org/wiki/LibreOffice_Calc>`__)
as well as astronomy table programs such as e.g.
`TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`__.
Since it's a simple text format it's easy to read or edit in any text editor or
put under version control (using e.g. `git <http://git-scm.com/>`__ or
`SVN <http://en.wikipedia.org/wiki/Apache_Subversion>`__).
CSV files are not standardised (there's many variants which causes problems in practice),
don't support metadata (e.g. units or descriptions of columns).

A `tabular data package format <http://dataprotocols.org/tabular-data-package/>`__ has
been defined with a clear CSV format specification and associated metadata in an extra JSON file
(see also `here <https://github.com/astropy/astropy-APEs/pull/7>`__).

To read and write CSV data from Python you can use the extensible `astropy.io.ascii` methods
via the `unified Astropy table I/O interface <http://docs.astropy.org/en/latest/io/unified.html>`__

.. code-block:: python

   from astropy.table import Table
   table = Table.read('measurements.csv', format='csv')
   table.write('measurements.tex', format='latex')

There's also the
`Python standard library csv module <http://pymotw.com/2/csv/>`__ as well as the
`numpy text I/O methods <http://docs.scipy.org/doc/numpy/reference/routines.io.html#text-files>`__ and the
`pandas text I/O methods <http://pandas.pydata.org/pandas-docs/stable/io.html>`__ ...
each have certain advantages / disadvantages (e.g. availability, features, speed).

JSON
++++

**JSON** files (see `Wikipedia <http://en.wikipedia.org/wiki/JSON>`__)

TODO: describe

XML
+++

**XML** files (see `Wikipedia <http://en.wikipedia.org/wiki/Xml>`__)

GammaLib / ctools uses an "observation definition" XML format described
`here <http://cta.irap.omp.eu/gammalib-devel/user_manual/modules/obs.html#describing-observations-using-xml>`__.

TODO: describe

FITS
++++

**FITS** files (see `Wikipedia <https://en.wikipedia.org/wiki/FITS>`__)

TODO: describe

ROOT
++++

**ROOT** files (see `Wikipedia <https://en.wikipedia.org/wiki/ROOT>`__)
This is a binary serialisation format (see `TFile <https://root.cern.ch/root/html/TFile.html>`__)
that is very common for low-level data in high-energy physics and astronomy and for
computing and storing instrument response functions.
If only ROOT built-in objects like simple `TTree <https://root.cern.ch/root/html/TTree.html>`__ and
`Histogram <https://root.cern.ch/root/html/TH1.html>`__  objects are stored it is
possible to exchange those files and read them from C++, Python (via `PyROOT`_ or `rootpy`_).
Access to your own serialised C++ objects is only possible if you distribute ROOT and
a C++ library ... but storing data this way is anyways a bad idea
(see e.g. `here <https://www.youtube.com/watch?v=7KnfGDajDQw>`__).

TODO: give examples how to read / convert ROOT data (e.g. to FITS).

Other
+++++

Other file formats that are very useful but not commonly used in gamma-ray astronomy (yet):

* **HDF5** files (see `Wikipedia <https://en.wikipedia.org/wiki/Hierarchical_Data_Format#HDF5>`__).
  Advantages over FITS: much faster for some applications,
  more flexible metadata, more widespread use (not astro specific),
  some tools for schema validation.
* There's a bunch of efficient and flexible binary data serialization formats, e.g.
  `Google Protobuf <https://code.google.com/p/protobuf/>`__ or
  `MessagePack <https://msgpack.org/>`__ or `BSON <https://bsonspec.org/>`__.

TODO: describe that most of these formats are only being considered for low-level data
for CTA, e.g. shower image I/O can be much more efficient that with FITS variable-length columns.

* Pickle: should never be used explicitly. But it is implicitly used by multiprocessing
  for communication with subprocesses, so if you use that you care if your objects can be
  pickled. (Do we care at all for Gammapy or is our policy that we don't support pickling
  Gammapy objects?)

* `SQLite <https://sqlite.org/>`__ gives you a `SQL <https://en.wikipedia.org/wiki/SQL>`__
  database in memory or a simple file (no server, no configuration).
  TODO: describe how it can be useful for pipeline processing (async I/O and easy select)


Validation
----------

What is it?
+++++++++++

When data and tools are deployed to users, it is necessary for the tools to validate the
input data and give good error messages when there is a problem.

The most common problems with user-edited input files
(e.g. INI config files or XML source model specifications or CSV runlists or ...)
is that the syntax is incorrect ... this will be noticed and reported by the
parser (e.g. a message like ``"expected '=' after element ABC on line XYZ"``).
It's usually out of your control and the error message is good enough for the
user to quickly find and fix the problem.

The second most common problem with user-edited input files is that the structure
or content doesn't match the specification.
Also format specifications change over time and there are tools that generate
output with incorrect structure or content, so this is not only an issue for user-generated files.

Checking the structure (and where possible content) is the responsibility of
the tool author and can be done either by writing a schema or code.
If you don't know what a schema is, please take a few minutes to read about it
`here <https://spacetelescope.github.io/understanding-json-schema/about.html>`__
using JSON as an example, I won't try to explain it here.

Existing Tools
++++++++++++++

TODO: Link collection and very short description of existing format and schema validation tools.

* ftverify

The following tools are available for schema validation of the file formats listed above
(with a strong emphasis on Python tools):

* INI
* CSV
* JSON
* XML
* FITS
* ROOT
* HDF5


`CSV schema <https://pypi.python.org/pypi/CsvSchema/>`__
use of such schemas

* http://embray.github.io/PyFITS/schema/users_guide/users_schema.html
* https://groups.google.com/d/msg/astropy-dev/CFGnVguRlgs/yObfzPTWvNkJ
* http://spacetelescope.github.io/understanding-json-schema/index.html

With Gammapy
++++++++++++

TODO: Implement ``gp_verify`` tool that can check the most common gamma-ray
data formats (e.g. event lists, ...).

Useful links
------------

* http://sedfitter.readthedocs.io/en/stable/creating_model_packages.html#sed-files
* http://fits.gsfc.nasa.gov/fits_registry.html

