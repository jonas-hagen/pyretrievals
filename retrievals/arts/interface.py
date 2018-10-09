import numpy as np
from scipy import sparse
from collections import namedtuple

from dotenv import load_dotenv

load_dotenv(dotenv_path='./.env')

from typhon.arts.workspace import Workspace, arts_agenda
from typhon.arts import xml
from typhon.arts.catalogues import Sparse

from retrievals.arts import boilerplate
from retrievals.arts import retrieval


def _is_asc(x):
    """Check if vector is strongly monotonic increasing."""
    return np.all(np.diff(x) > 0)


def _is_desc(x):
    """Check if vector is strongly monotonic decreasing."""
    return np.all(np.diff(x) < 0)


Observation = namedtuple('Observation', ['time', 'za', 'aa', 'lat', 'lon', 'alt'])


class ArtsController:
    """The equivalent for a cfile."""

    def __init__(self):
        self.ws = Workspace()
        self.retrieval_quantities = []

    def setup(self, atmosphere_dim=1, iy_unit='RJBT', ppath_lmax=-1, stokes_dim=1):
        """
        Run boilerplate (includes, agendas) and set basic variables.
        :param atmosphere_dim:
        :param iy_unit:
        :param ppath_lmax:
        :param stokes_dim:
        """
        boilerplate.include_general(self.ws)
        boilerplate.copy_agendas(self.ws)
        boilerplate.set_basics(self.ws, atmosphere_dim, iy_unit, ppath_lmax, stokes_dim)

        # Deactivate some stuff (can be activated later)
        self.ws.jacobianOff()
        self.ws.cloudboxOff()

    def checked_calc(self, negative_vmr_ok=False):
        """Run checked calculations."""
        boilerplate.run_checks(self.ws, negative_vmr_ok)

    def set_spectroscopy(self, abs_lines, abs_species, line_shape=None, f_abs_interp_order=3):
        """
        Setup absorption species and spectroscopy data.

        :param ws: The workspace.
        :param abs_lines: Absoption lines.
        :param abs_species: List of abs species tags.
        :param line_shape: Line shape definition. Default: ['Voigt_Kuntz6', 'VVH', 750e9]
        :param f_abs_interp_order: No effect for OnTheFly propmat. Default: 3
        :type abs_lines: typhon.arts.catalogues.ArrayOfLineRecord
        """
        boilerplate.setup_spectroscopy(self.ws, abs_lines, abs_species, line_shape)
        self.ws.f_abs_interp_order = f_abs_interp_order  # no effect for OnTheFly propmat

    def set_spectroscopy_from_file(self, abs_lines_file, abs_species, line_shape=None):
        """
        Setup absorption species and spectroscopy data from XML file.

        :param ws: The workspace.
        :param abs_lines_file: Path to an XML file.
        :param abs_species: List of abs species tags.
        :param line_shape: Line shape definition. Default: ['Voigt_Kuntz6', 'VVH', 750e9]
        """
        abs_lines = xml.load(abs_lines_file)
        self.set_spectroscopy(abs_lines, abs_species, line_shape)

    def set_grids(self, f_grid, p_grid, lat_grid=None, lon_grid=None):
        """
        Set the forward model grids. Basic checks are performed, depending on dimensionality of atmosphere.

        :param f_grid:
        :param p_grid:
        :param lat_grid:
        :param lon_grid:
        """
        if not self.ws.atmosphere_dim.initialized:
            raise Exception('atmosphere_dim must be initialized before assigning grids.')
        if not _is_asc(f_grid):
            raise ValueError('Values of f_grid must be strictly increasing.')
        if not _is_desc(p_grid) or not np.all(p_grid > 0):
            raise ValueError('Values of p_grid must be strictly decreasing and positive.')
        if self.ws.atmosphere_dim.value == 1:
            if lat_grid is not None or lon_grid is not None:
                raise ValueError('For 1D atmosphere, lat_grid and lon_grid shall be empty.')
        elif self.ws.atmosphere_dim.value == 2:
            if lon_grid is not None or len(lat_grid):
                raise ValueError('For 2D atmosphere, lon_grid shall be empty.')
            if lat_grid is None or len(lat_grid) == 0:
                raise ValueError('For 2D atmosphere, lat_grid must be set.')
        elif self.ws.atmosphere_dim.value == 3:
            if lat_grid is None or len(lat_grid) < 2 or lon_grid is None or len(lon_grid) < 2:
                raise ValueError('For 3D atmosphere, length of lat_grid and lon_grid must be >= 2.')
            if max(abs(lon_grid)) > 360:
                raise ValueError('Values of lon_grid must be in the range [-360,360].')
            if max(abs(lat_grid)) > 90:
                raise ValueError('Values of lat_grid must be in the range [-90,90].')
        if lat_grid is not None and not _is_asc(lat_grid):
            raise ValueError('Values of lat_grid must be strictly increasing.')
        if lon_grid is not None and not _is_asc(lon_grid):
            raise ValueError('Values of lon_grid must be strictly increasing.')

        self.ws.f_grid = f_grid
        self.ws.p_grid = p_grid
        self.ws.lat_grid = lat_grid
        self.ws.lon_grid = lon_grid
        self.set_surface(0)

    def set_surface(self, altitude: float):
        """
        Set surface altitude.

        :param float altitude:

        .. note:: Currently, only constant altitudes are supported.
        """
        self.ws.z_surface = altitude * np.ones((self.n_lat, self.n_lat))

    def set_atmosphere(self, atmosphere):
        """
        Set the atmospheric state.

        :param atmosphere: Atmosphere with Temperature, Altitude and VMR Fields.
        :type atmosphere: retrievals.arts.atmosphere.Atmosphere

        .. note:: Currently only supports 1D atmospheres that is then expanded to a
            multi-dimensional homogeneous atmosphere.
        """

        self.ws.t_field_raw = atmosphere.t_field
        self.ws.z_field_raw = atmosphere.z_field
        self.ws.vmr_field_raw = [atmosphere.vmr_field(mt) for mt in self.abs_species_maintags]
        self.ws.nlte_field_raw = None

        if atmosphere.dimensions == 1:
            self.ws.AtmFieldsCalcExpand1D()
        else:
            raise NotImplementedError(
                'Inhomogeneous (multi-dimensional) raw atmospheres are not implemented (and not tested).')
            # ws.AtmFieldsCalc()  Maybe?

    def set_wind(self, atmosphere=None, components=None):
        """
        Set the wind fields. If no atmosphere is supplied, all wind fields are set to zero.

        :param atmosphere: Atmosphere with Wind fields. If None, set wind to zero.
        :type atmosphere: retrievals.arts.atmosphere.Atmosphere
        :param components: Tuple with elements 'u', 'v' and/or 'w'. Default: ('u', 'v')

        .. note:: Currently only supports 1D atmospheres that is then expanded to a
            multi-dimensional homogeneous atmosphere.
        """

        if atmosphere is None:
            self.ws.u_field = []
            self.ws.v_field = []
            self.ws.w_field = []
            return

        if components is None:
            components = ('u', 'v')

        if 'u' in components:
            self.ws.u_field_raw = atmosphere.wind_field('u')
        if 'v' in components:
            self.ws.v_field_raw = atmosphere.wind_field('v')
        if 'w' in components:
            self.ws.w_field_raw = atmosphere.wind_field('w')

        if atmosphere.dimensions == 1:
            self.ws.WindFieldsCalcExpand1D()
        else:
            raise NotImplementedError(
                'Inhomogeneous (multi-dimensional) raw atmospheres are not implemented (and not tested).')
            # ws.AtmFieldsCalc()  Maybe?

    def set_observations(self, observations):
        """
        Set the geometry of the observations made.

        :param observations:
        :type observations: Iterable[retrievals.arts.interface.Observation]
        """
        self.ws.sensor_time = np.array([obs.time for obs in observations])
        self.ws.sensor_los = np.array([[obs.za, obs.aa] for obs in observations])
        self.ws.sensor_pos = np.array([[obs.alt, obs.lat, obs.lon] for obs in observations])

    def set_y(self, ys):
        y = np.concatenate(ys)
        self.ws.y = y

    def set_sensor(self, sensor):
        """
        Set the sensor.
        :param sensor:
        :type sensor: retrievals.arts.sensors.AbstractSensor
        """
        sensor.apply(self.ws)

    def y_calc(self, jacobian_do=False):
        """
        Run the forward model.
        :param jacobian_do: Not implemented yet.
        :return: The measurements as list with length according to observations.
        """
        if jacobian_do:
            raise NotImplementedError('Jacobian not implemented yet.')
            # self.ws.jacobian_do = 1
        self.ws.yCalc()
        y = np.copy(self.ws.y.value)
        return np.split(y, self.n_obs)

    def define_retrieval(self, retrieval_quantities, y_vars):
        """
        Define the retrieval quantities.
        :param retrieval_quantities: Iterable of retrieval quantities `retrievals.arts.retrieval.RetrievalQuantity`.
        :param y_vars: Variance of the measurement noise.
        """
        ws = self.ws

        if len(y_vars) != self.n_y:
            raise ValueError('Variance vector y_vars must have same length as y.')
        ws.retrievalDefInit()

        # Retrieval quantities
        self.retrieval_quantities = retrieval_quantities
        for rq in retrieval_quantities:
            rq.apply(ws)

        # Se and its inverse
        covmat_block = sparse.diags(y_vars, format='csr')
        boilerplate.set_variable_by_xml(ws, ws.covmat_block, covmat_block)
        ws.covmat_seAddBlock(block=ws.covmat_block)

        ws.retrievalDefClose()

    def oem(self, method='li', max_iter=10, stop_dx=0.001, lm_ga_settings=None, display_progress=True,
            inversion_iterate_agenda=None):
        """
        Run the optimal estimation. See Arts documentation for details.
        :param method:
        :param max_iter:
        :param stop_dx:
        :param lm_ga_settings: Default: [10, 2, 2, 100, 1, 99]
        :param display_progress:
        :param inversion_iterate_agenda: If set to None, a simple default agenda is used.
        :return:
        """
        ws = self.ws

        if lm_ga_settings is None:
            lm_ga_settings = [10, 2, 2, 100, 1, 99]
        lm_ga_settings = np.array(lm_ga_settings)

        if inversion_iterate_agenda is None:
            inversion_iterate_agenda = boilerplate.inversion_iterate_agenda

        ws.Copy(ws.inversion_iterate_agenda, inversion_iterate_agenda)

        if ws.dxdy.initialized:
            ws.Delete(ws.dxdy)  # This is a workaround to see if OEM converged

        ws.xaStandard()
        ws.OEM(method=method, max_iter=max_iter, stop_dx=stop_dx, lm_ga_settings=lm_ga_settings,
               display_progress=1 if display_progress else 0)

        if self.oem_converged:  # Just checks if dxdy is initialized
            ws.avkCalc()
            ws.covmat_ssCalc()
            ws.covmat_soCalc()
            ws.retrievalErrorsExtract()

            x = ws.x.value
            avk = ws.avk.value
            eo = ws.retrieval_eo.value
            es = ws.retrieval_eo.value

            for rq in self.retrieval_quantities:
                rq.extract_result(x, avk, eo, es)

        return self.oem_converged

    @property
    def p_grid(self):
        return self.ws.p_grid.value

    @property
    def lat_grid(self):
        return self.ws.lat_grid.value

    @property
    def lon_grid(self):
        return self.ws.lon_grid.value

    @property
    def n_lat(self):
        return len(self.ws.lat_grid.value)

    @property
    def n_lon(self):
        return len(self.ws.lat_grid.value)

    @property
    def n_y(self):
        return len(self.ws.y.value)

    @property
    def n_obs(self):
        if self.ws.sensor_time.initialized:
            return len(self.ws.sensor_time.value)
        else:
            return 0

    @property
    def abs_species_maintags(self):
        abs_species = self.ws.abs_species.value
        maintags = [st[0].split('-')[0] for st in abs_species]
        return maintags

    @property
    def oem_converged(self):
        return self.ws.dxdy.initialized and self.ws.dxdy.value is not None and self.ws.dxdy.value.shape != (0, 0)

