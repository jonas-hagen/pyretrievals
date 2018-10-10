Quickstart
==========



The interface to ARTS
---------------------

The :py:class:`ArtsController` is aimed to be a simple, pythonic substitution for the ARTS controlfiles.

A simple example for a simulated measurement of an Ozone spectrometer::

    from retrievals import arts

    ac = arts.ArtsController()
    ac.setup(atmosphere_dim=3)
    ac.set_spectroscopy_from_file(
        os.path.join(TEST_DATA_PATH, 'Perrin_O3_142.xml'),
        ['O3', ],
    )
    ac.set_grids(
        f_grid=np.linspace(-500e6, 500e6, 300) + f0,
        p_grid=np.logspace(5, -1, 100),
        lat_grid=np.linspace(-4, 4, 5),
        lon_grid=np.linspace(-4, 4, 5)
    )
    ac.set_surface(1e3)

    # Load Atmosphere from climatology in arts-xml-data
    basename = os.path.join(TEST_DATA_PATH, 'fascod-tropical/tropical.')
    atm = arts.Atmosphere.from_arts_xml(basename)
    ac.set_atmosphere(atm)

    # Set the backend
    f_backend = np.linspace(-400e6, 400e6, 600) + f0
    ac.set_sensor(arts.SensorGaussian(f_backend, np.array([800e6/600, ])))

    # Set east- and westward Observations
    ac.set_observations([arts.Observation(
                            time=0, lat=0, lon=0, alt=12e3,
                            za=90 - 22, aa=azimuth)
                         for azimuth in [90, -90]])
    ac.checked_calc()

    # Run forward model
    y_east, y_west = ac.y_calc()

Now we have the simulated spectra for the two Observations in `y_east` and `y_west` respectively.