ARTS
====

.. automodule:: retrievals.arts

Summary
-------

General ARTS Interface
''''''''''''''''''''''

.. autosummary::
    :nosignatures:

    ArtsController
    Observation
    OemException
    Atmosphere


Sensor definitions
''''''''''''''''''

.. autosummary::
    :nosignatures:

    SensorOff
    SensorGaussian
    SensorFFT


Retrieval Quantities
''''''''''''''''''''

.. autosummary::
    :nosignatures:

    RetrievalQuantity
    GriddedRetrievalQuantity
    AbsSpecies
    Wind
    FreqShift
    Polyfit


Reference
---------

.. autoclass:: ArtsController

    .. automethod:: setup
    .. automethod:: set_grids
    .. automethod:: set_surface
    .. automethod:: set_spectroscopy
    .. automethod:: set_spectroscopy_from_file
    .. automethod:: set_atmosphere
    .. automethod:: apply_hse
    .. automethod:: set_wind
    .. automethod:: set_sensor
    .. automethod:: set_observations
    .. automethod:: checked_calc
    .. automethod:: y_calc
    .. automethod:: set_y
    .. automethod:: define_retrieval
    .. automethod:: oem
    .. automethod:: get_level2_xarray

.. autoclass:: Observation

    .. autoattribute:: za
    .. autoattribute:: aa
    .. autoattribute:: lat
    .. autoattribute:: lon
    .. autoattribute:: alt
    .. autoattribute:: time

.. autoclass:: OemException
    :members:

.. autoclass:: Atmosphere
    :members:

.. autoclass:: SensorOff
    :members:

.. autoclass:: SensorGaussian
    :members:

.. autoclass:: SensorFFT
    :members:

.. autoclass:: RetrievalQuantity
    :members:

.. autoclass:: GriddedRetrievalQuantity
    :members:

.. autoclass:: Wind
    :members:

.. autoclass:: FreqShift
    :members:

.. autoclass:: AbsSpecies
    :members:

.. autoclass:: Polyfit
    :members:
