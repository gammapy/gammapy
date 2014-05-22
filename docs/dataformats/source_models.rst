.. _dataformats_source_models:

Source models
=============

TODO: Explain why there are different formats and give converter functions / tools. 

XML file format
---------------

GammaLib / ctools uses an "model definition" XML format described
`here <http://gammalib.sourceforge.net/user_manual/modules/model.html#overview>`__

The Fermi ``gtlike`` tool uses the same format (the implemented models are a bit different) described
`here <http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html>`__

The same format is used by ``ctlike`` and ``gtlike`` to report the model fit results,
which has the serious draw-back that useful information like the fit covariance matrix, asymmetric errors
or numbers like the fit statistic or predicted number of counts are not output
in a machine-readable format. 

Configuration format
--------------------

TODO: document


JSON format
-----------

TODO: document via a JSON schema.
