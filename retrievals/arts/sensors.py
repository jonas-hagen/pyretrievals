from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np

from typhon.arts.workspace import arts_agenda
from typhon.arts.griddedfield import GriddedField1


class AbstractSensor(ABC):
    """
    Abstract Sensor requires the implementation of an `sensor_response_agenda` property.
    See the specific derived classes for how to use a sensor.
    """

    def apply(self, ws):
        """Copy and execute sensor response agenda."""
        ws.Copy(ws.sensor_response_agenda, self.sensor_response_agenda)
        ws.AgendaExecute(ws.sensor_response_agenda)

    @property
    @abstractmethod
    def sensor_response_agenda(self):
        """
        The sensor response agenda.

        :type: typhon.arts.workspace.agendas.Agenda

        .. seealso:: Decorator :py:func:`typhon.arts.workspace.workspace.arts_agenda`.
        """
        pass

    @property
    @abstractmethod
    def f_backend(self):
        pass


class SensorOff(AbstractSensor):
    """Sensor that does nothing."""

    def __init__(self):
        pass

    def apply(self, ws):
        """Copy and execute sensor response agenda."""
        ws.AntennaOff()
        ws.sensorOff()
        pass

    @property
    def sensor_response_agenda(self):
        return None

    @property
    def f_backend(self):
        return None


class SensorFFT(AbstractSensor):
    """
    Sensor with channel response for an FFT Spectrometer with :math:`\mathrm{sinc}^2` response.
    """

    def __init__(self, f_backend, resolution, num_channels=10):
        """
        :param f_backend: The backend frequency vector.
        :param resolution: The frequency resolution of the FFTS in Hz
        :param num_channels: Number of channels with nonzero response, default: 10
        """
        self._f_backend = f_backend
        self.resolution = resolution
        self.num_channels = num_channels

        # Compute the backend channel response
        grid = np.linspace(-self.num_channels / 2,
                           self.num_channels / 2,
                           20 * self.num_channels)
        response = np.sinc(grid) ** 2
        self.bcr = GriddedField1(name='Backend channel response function for FFTS',
                                 gridnames=['Frequency'], dataname='Data',
                                 grids=[self.resolution * grid],
                                 data=response)

    def apply(self, ws):
        # Modify workspace
        ws.FlagOn(ws.sensor_norm)
        ws.f_backend = self.f_backend
        ws.backend_channel_response = [self.bcr, ]

        super().apply(ws)

    @property
    def sensor_response_agenda(self):
        @arts_agenda
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.sensor_responseInit()
            ws.sensor_responseBackend()

        return sensor_response_agenda

    @property
    def f_backend(self):
        return self._f_backend


class SensorGaussian(AbstractSensor):
    """Sensor with Gaussian Channel response."""

    def __init__(self, f_backend, fwhm):
        """
        :param f_backend: Backend frequencies
        :param fwhm: Full width at half maximum (resolution)
        """
        self._f_backend = f_backend
        self.fwhm = fwhm

    def apply(self, ws):
        ws.FlagOn(ws.sensor_norm)
        ws.f_backend = self.f_backend
        ws.backend_channel_responseGaussian(fwhm=self.fwhm)
        super().apply(ws)

    @property
    def sensor_response_agenda(self):
        @arts_agenda
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.sensor_responseInit()
            ws.sensor_responseBackend()

        return sensor_response_agenda

    @property
    def f_backend(self):
        return self._f_backend
