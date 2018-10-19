Examples
========


Example 1, `test_oem.py`
------------------------

.. autofunction:: examples.test_oem.main

See the file ``examples/test_oem.py`` which is very close to the ``controlfiles/artscomponents/test_oem.arts`` cfile shipped with arts.
It simulates the ozone line at 110 GHz and retrieves ozone VMR, frequency shift and a polynomial baseline.


To run it, make sure that the ``ARTS_BUILD_PATH`` and ``ARTS_DATA_PATH`` is set in your environment.
It also plots the retrieval results and saves the results to a netcdf file.

The output looks like this:

.. code-block:: none

    Loading ARTS API from: /opt/arts-dev/arts/build/src/libarts_api.so

                                    MAP Computation
    Formulation: Standard
    Method:      Gauss-Newton

     Step     Total Cost         x-Cost         y-Cost    Conv. Crit.
    --------------------------------------------------------------------------------
        0        254.622              0        254.622
        1      0.0139533     0.00872973     0.00522353        8486.95
        2     0.00897012     0.00873093     0.00023919       0.167512
        3     0.00896818     0.00875997    0.000208204    7.83861e-05
    --------------------------------------------------------------------------------

    Total number of steps:            3
    Final scaled cost function value: 0.00896818
    OEM computation converged.

    Elapsed Time for Retrieval:                       1.82412
    Time in inversion_iterate Agenda (No Jacobian):   0.343551
    Time in inversion_iterate Agenda (With Jacobian): 1.46146

                                          ----

    Fshift fit: 148.675 kHz, true: -150 kHz
    Poly coefficients: 1.97, 1.00 true: 2, 1

    Saved plots to test_oem_*.png

    Saved results to test_oem_result.nc



.. image:: _static/test_oem_spectrum.*
    :width: 60%
    :align: center

.. image:: _static/test_oem_ozone.*
    :width: 60%
    :align: center
