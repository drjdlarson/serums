serums
======

A Python package for Statistical Error and Risk Utility for Multi-sensor Systems (SERUMS) developed by the Laboratory for Autonomy, GNC, and Estimation Research (LAGER) at the University of Alabama (UA).

.. contents:: Table of Contents
    :depth: 2
    :local:


..
    BEGIN TOOLCHAIN INCLUDE

.. _serums: https://github.com/drjdlarson/serums
.. _STACKOVERFLOW: https://stackoverflow.com/questions/69704561/cannot-update-spyder-5-1-5-on-new-anaconda-install
.. _SUBMODULE: https://git-scm.com/book/en/v2/Git-Tools-Submodules
.. |Open in Dev Containers| image:: https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode
   :target: https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/drjdlarson/serums.git



Setup and Installation
----------------------
Currently this package is not available via pip, this provides a guide on installation from the git repository both for developing for the package and for using the package with other code. Note that Python 3 is required, pytest is used for managing the built-in tests, and tox is used for automating the testing and documentation generation.


Installation for using the package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use this package when developing other software, simply clone the repository (or otherwise download the code). Then navigate in a terminal to the directory where the code was saved. Activate any python virtual environment if desired. Then :code:`pip install -e .` if pip maps to your Python 3 installation. The :code:`-e` option installs it as an editable package so if you do a git pull, you won't have to reinstall serums.


Installation for Developing for SERUMS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The fastest way to get started with development is to use VS Code with the dev container extension which uses docker. All you need to do is have VS Code and docker installed then click on the badge |Open in Dev Containers|. VS Code has instructions for getting started with dev containers `here <https://code.visualstudio.com/docs/devcontainers/containers>`_. If you are on windows you will need to enable WSL within windows and install the WSL extension within VS Code. 

Alternatively, the repository can be cloned locally, opened in VS Code, and then the container started with your local directory mounted within the container. This can make it easier to manage the files since they are controlled by your host OS instead of hidden within a docker volume. In this case, clone the repository like normal and open the root level folder within VS Code. If you have the dev container extension installed you will be prompted to reopen the folder in the container, select yes and the toolchain will automatically be installed for you. If you are on Windows you will first need to connect to the WSL session, then you can open the folder and will be prompted to reopen it in a container. Additionally, for Windows, it is recommended to keep the code within the home folder of the WSL system as this increases the performance when using containers and VS Code. To access the WSL directories from your file browser navigate to :code:`\\wsl$` and select your linux distribution.

You can also skip VS Code and use the container directly if desired. Or download the repository and work with a local python installation in another IDE.


Testing
-------
Unit and validation tests make use of **pytest** for the test runner, and tox for automation. The test scripts are located within the **test/** sub-directory.
The tests can be run through a command line with python 3 and tox installed. The tests can also be run as standalone scripts from command line by uncommenting the appropriate line under the :code:`__main__` section.

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
You can list the available environments within tox by running

.. code-block:: bash

    tox -av

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
    author       = {Jordan D. Larson, et al.},
    howpublished = {Web page},
    title        = {{SERUMS}: A Python library for Statistical Error and Risk Utility for Multi-sensor Systems},
    year         = {2022},
    url          = {https://github.com/drjdlarson/serums},
    }
