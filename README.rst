serums
======

A Python package for Statistical Error and Risk Utility for Multi-sensor Systems (SERUMS) developed by the Laboratory for Autonomy, GNC, and Estimation Research (LAGER) at the University of Alabama (UA).

..
    BEGIN TOOLCHAIN INCLUDE

.. _serums: https://github.com/drjdlarson/serums
.. _STACKOVERFLOW: https://stackoverflow.com/questions/69704561/cannot-update-spyder-5-1-5-on-new-anaconda-install
.. _SUBMODULE: https://git-scm.com/book/en/v2/Git-Tools-Submodules


Setup
-----
Currently this package is not available via pip, this provides a guide on installation from the git repository. Note that Python 3 is required, pytest is used for managing the built-in tests, and tox is used for automating the testing and documentation generation. It is recommended to use Anaconda and the Spyder IDE, but this is not necessary. The general process follows the following order with more details in the corresponding subsections.

#. Install Anaconda and Spyder and update to the latest version *[Optional]*. See `Installing and Updating Spyder`_.
#. Clone serums.
#. Install serums to the base conda environment. See `Installing serums`_.

    * Other virtual environments can be used for development/projects but it seems serums is needed in the base to get Spyder's variable explorer to work with its custom types.

If using Anaconda and Spyder you should be able to run tests with tox, generate documentation with tox, and run tests as standalone scripts from within Spyder (See `Testing`_ and `Building Documentation`_). Also the Spyder IDE variable explorer should recognize serums data types for debugging. It is also possible to create conda environments and tell Spyder to use that as the interpreter for running code. Note that the proper version of :code:`spyder-kernels` must be installed in the environment but Spyder will tell you the command to run when it fails to start the interpreter. This can be useful if you need additional libraries for certain projects. Information on conda environments can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ and setting up Spyder with a custom interpreter can be done through Spyder's settings.


Installing and Updating Spyder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Download and install `Anaconda <https://www.anaconda.com/>`_. This should also install Spyder, numpy, matplotlib, and some other common libraries to a base conda (virtual) environment.
#. Open Spyder and check the version by going to about spyder from the help menu.

    * If it is not version >= 5.1.5 and you want to update to the version 5.1.5 (recommended), close Spyder and run the following commands from a terminal (Anaconda prompt on windows) the second and third commands may give errors but they can be ignored. See here on `stackoverflow`_.

        .. code-block:: bash

            conda remove spyder
            conda remove python-language-server
            conda update anaconda
            conda install spyder=5.1.5


Installing serums
^^^^^^^^^^^^^^^^^
#. Download/clone the `serums`_ repository and save it somewhere on your system (remember the location).
#. Open the terminal that has the base Anaconda environment activated (normal terminal for linux, Anaconda prompt on windows).
#. Navigate to the directory where you saved the repository.

    * The directory structure should look like the following.

        ::

            . (YOU ARE HERE)
            └── serums/
                ├── setup.py
                └── serums


#. Install serums.

    * If using anaconda then install without dependencies to allow conda to manage dependencies. Note, :code:`PATH_TO_SERUMS` is :code:`./serums/` if following the recommended steps.

        .. code-block:: bash

            conda install numpy scipy matplotlib
            pip install --no-dependencies -e PATH_TO_SERUMS

    * If not using anaconda then serums can be installed with the dependencies. Note, :code:`PATH_TO_SERUMS` follows the format in the above bullet.

        .. code-block:: bash

            pip install -e PATH_TO_SERUMS

#. If using Anaconda, then to run the built-in tests as standalone scripts, install the test dependencies *[Optional]*.

    .. code-block:: bash

        conda install pytest

#. Install tox for automated testing and building the documentation *[Optional]*

    * For Anaconda run

    .. code-block:: bash

        conda install -c conda-forge tox

    * If not using Anaconda then run

    .. code-block:: bash

        pip install tox


Testing
-------
Unit and validation tests make use of **pytest** for the test runner, and tox for automation. The test scripts are located within the **test/** sub-directory.
The tests can be run through a command line with python 3 and tox installed. If the Spyder setup instructions were followed then the tests can also be run as standalone scripts from within Spyder by uncommenting the appropriate line under the :code:`__main__` section.

There are 3 different environments for running tests. One for unit tests, another for validation tests, and a general purpose one that accepts any arguments to pytest.
The general purpose environment is executed by running

.. code-block:: bash

    tox -e test -- PY_TEST_ARGS

where :code:`PY_TEST_ARGS` are any arguments to be passed directly to the pytest command (Note: if none are passed the :code:`--` is not needed).
For example to run any test cases containing a keyword, run the following,

.. code-block:: bash

    tox -e test -- -k guidance

To run tests marked as slow, pass the :code:`--runslow` option.

The unit test environment runs all tests within the **test/unit/** sub-directory. These tests are designed to confirm basic functionality.
Many of them do not ensure algorithm performance but may do some basic checking of a few key parameters. This environment is run by

.. code-block:: bash

    tox -e unit_test -- PY_TEST_ARGS

The validation test environment runs all tests within the **test/validation/** sub-directory. These are designed to verify algorithm performance and include more extensive checking of the output arguments against known values. They often run slower than unit tests.
These can be run with

.. code-block:: bash

    tox -e validation_test -- PY_TEST_ARGS


Building Documentation
----------------------
The documentation uses sphinx and autodoc to pull docstrings from the code. This process is run through a command line that has python 3 and tox installed. The built documentation is in the **docs/build/** sub-directory.
The HTML version of the docs can be built using the following command

.. code-block:: bash

    tox -e docs -- html

Then they can be viewed by opening **docs/build/html/index.html** with a web browser.


Notes about tox
---------------
If tox is failing to install the dependencies due to an error in distutils, then it may be required to instal distutils seperately by

.. code-block:: bash

    sudo apt install python3.7-distutils

for a debian based system.

..
    END TOOLCHAIN INCLUDE

Cite
====
Please cite the framework as follows

.. code-block:: bibtex

    @Misc{serums,
    author       = {Jordan D. Larson and Ryan W. Thomas and Vaughn Weirens},
    howpublished = {Web page},
    title        = {{SERUMS}: A Python library for Statistical Error and Risk Utility for Multi-sensor Systems},
    year         = {2022},
    url          = {https://github.com/drjdlarson/serums},
    }
