Data Fusion Examples
====================

.. contents:: Examples
   :depth: 2
   :local:



Gaussian Multiplication
-----------------------
The PDF of the multiplication of two zero mean normal PDFs follows a Normal Product distribution
see `here <https://mathworld.wolfram.com/NormalProductDistribution.html>`_. 
For random variables :math:`X, Y` with zero mean and standard deviations :math:`\sigma_X, \sigma_Y` the
product of their PDF follows the PDF given by

.. math:: 

   P_{X,Y}(u) = \frac{K_0(\frac{\vert u \vert}{\sigma_X\sigma_Y})}{\pi\sigma_X\sigma_Y}

where :math:`K_0(z)` is the modified Bessel function of the second kind.

.. literalinclude:: /example_scripts/distribution_overbounder/fusion_gaussian.py
   :linenos:
   :pyobject: multiplication

The above script gives this as output.

.. image:: /example_scripts/distribution_overbounder/fusion_gaussian_multiplication.png
   :align: center


General Polynomial
------------------
Data fusion for a generic polynomial function can be performed as shown in the following script. Note Gaussians are used here
as an example however, any child of :class:`serums.models.BaseSingleModel` can be used.

.. literalinclude:: /example_scripts/distribution_overbounder/fusion_gaussian.py
   :linenos:
   :pyobject: main

The above script gives this as output.

.. image:: /example_scripts/distribution_overbounder/fusion_gaussian.png
   :align: center
