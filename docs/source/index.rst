Welcome to revertex's documentation!
====================================

_revertex_ is a python package to generate event vertices for _remage_ see [link](https://remage.readthedocs.io/en/latest).

The package contains tools to generate a variety of different vertices both with dedicated python functions,
and on the command line, following the specification defined [here](https://remage.readthedocs.io/en/latest/manual/generators.html#simulating-event-vertices-and-kinematics-from-external-files)


Getting started
---------------

The package can currently only be installed with git.

```console
$ git clone git@github.com:legend-exp/revertex.git
$ cd revertex
$ pip install .
```

Now you are ready to generate some event vertices. For most cases this can be
done using the command line:

```console

$ revertex -h

```
and then following the instructions to generate vertices for a specific case.

If you want to use _revertex_ tools to generate vertices directly in python code see the next section!

Generator specification (_advanced_)
------------------------------------

The package consists of some "core" functionality, for example to format the output
correctly for _remage_ and also a set of specific generators.
For these functions we use the following specification:

- A position generator should be written as a function with ``size`` and ``seed`` as required arguments,
- All other options should be given by keyword arguments
- The code should return a 2D numpy array of each generated ``(x,y,z)`` position.


Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   Package API reference <api/modules>
