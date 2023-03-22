Data Fusion Examples
====================

.. contents:: Examples
   :depth: 2
   :local:


Gaussian Polynomial
-------------------
Data fusion for a generic polynomial function can be performed as shown in the following script. Note Gaussians are used here
as an example however, any child of :class:`serums.models.BaseSingleModel` can be used.

.. literalinclude:: /example_scripts/distribution_overbounder/fusion_gaussian.py
   :linenos:
   :pyobject: main

The above script gives this as output.

.. image:: /example_scripts/distribution_overbounder/fusion_gaussian.png
   :align: center
