Installation
============

To test changes in the package, you install it in `editable mode`_ locally in
your virtualenv by running::

    $ make dev

This will also install our pre-commit hooks and local packages needed for style
checks.

.. tip::
    If you need to install a locally edited version of emmental in a separate location,
    such as an application, you can directly install your locally modified version::

        $ pip install -e path/to/emmental/

    in the virtualenv of your application.

.. _emmental/\_version.py: https://github.com/SenWu/emmental/blob/master/emmental/_version.py
.. _editable mode: https://packaging.python.org/tutorials/distributing-packages/#working-in-development-mode
