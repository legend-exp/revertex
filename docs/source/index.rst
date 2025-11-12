Welcome to revertex's documentation!
====================================

*revertex* is a python package to generate event vertices for *remage* see  `link <https://remage.readthedocs.io/en/latest>`_
The package contains tools to generate a variety of different vertices both with dedicated python functions,
and on the command line, following the specification defined `here <https://remage.readthedocs.io/en/latest/manual/generators.html#simulating-event-vertices-and-kinematics-from-external-files>`_


Getting started
---------------

The package can currently only be installed with git.

.. code-block:: console

   git clone git@github.com:legend-exp/revertex.git
   cd revertex
   pip install .


Now you are ready to generate some event vertices. For most cases this can be
done using the command line:

.. code-block:: console

   revertex -h

and then following the instructions to generate vertices for a specific case.

If you want to use *revertex* tools to generate vertices directly in python code see the next section!

Generator specification (*advanced*)
------------------------------------

The package consists of some "core" functionality, for example to format the output
correctly for *remage* and also a set of specific generators.
For these functions we use the following specification:

- A position generator should be written as a function with ``size`` and ``seed`` as required arguments,
- All other options should be given by keyword arguments
- The code should return a 2D numpy array of each generated ``(x,y,z)`` position.

This allows the "core" functionality of *revertex* to easily handle generation of the vertices in chunks, and
conversion to the correct output format.

More details
------------

.. toctree::
   :maxdepth: 1

   Package API reference <api/modules>
