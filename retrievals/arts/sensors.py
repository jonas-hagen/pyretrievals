from abc import ABC, abstractmethod
import numpy as np

from typhon.arts.workspace import arts_agenda
from typhon.arts.griddedfield import GriddedField1


class AbstractSensor(ABC):
    @abstractmethod
    def apply(self, ws):
        pass


class SensorOff(AbstractSensor):
    def __init__(self):
        pass

    def apply(self, ws):
        ws.AntennaOff()
        ws.SensorOff()


class SensorFFT(AbstractSensor):
    """
    Sensor with channel response for an FFT Spectrometer with sinc^2 response.
    """

    def __init__(self, f_backend, resolution, num_channels=10):
        """
        :param resolution: The frequency resolution of the FFTS in Hz
        :param num_channels: Number of channels with nonzero response, default: 10
        """
        self.f_backend = f_backend
        self.resolution = resolution
        self.num_channels = num_channels

    def apply(self, ws):
        grid = np.linspace(-self.num_channels / 2,
                           self.num_channels / 2,
                           20 * self.num_channels)
        response = np.sinc(grid) ** 2
        bcr = GriddedField1(name='Backend channel response function for FFTS',
                            gridnames=['Frequency'], dataname='Data',
                            grids=[self.resolution * grid],
                            data=response)

        # Modify workspace
        ws.FlagOn(ws.sensor_norm)
        ws.f_backend = self.f_backend
        ws.backend_channel_response = [bcr, ]

        @arts_agenda
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.sensor_responseInit()
            ws.sensor_responseBackend()

        ws.Copy(ws.sensor_response_agenda, sensor_response_agenda)
        ws.AgendaExecute(ws.sensor_response_agena)


class SensorGaussian(AbstractSensor):
    def __init__(self, f_backend, fwhm):
        self.f_backend = f_backend
        self.fwhm = fwhm

    def apply(self, ws):
        ws.FlagOn(ws.sensor_norm)
        ws.f_backend = self.f_backend
        ws.backend_channel_responseGaussian(fwhm=self.fwhm)

        @arts_agenda
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.sensor_responseInit()
            ws.sensor_responseBackend()

        ws.Copy(ws.sensor_response_agenda, sensor_response_agenda)
        ws.AgendaExecute(ws.sensor_response_agenda)
