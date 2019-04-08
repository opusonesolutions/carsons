import pint
from numpy import array
from numpy.testing import assert_array_almost_equal

from carsons.carsons import ConcentricNeutralCarsonsEquations
from tests.test_carsons import OHM_PER_MILE_TO_OHM_PER_METER

ureg = pint.UnitRegistry()

feet = ureg.feet
inches = ureg.inches
miles = ureg.miles
ohms = ureg.ohms


class ABCCable:
    """ Modelled after Kerstings 'Distribution System Modeling and Analysis'
        pg. 102 example """

    @property
    def neutral_strand_gmr(self):
        return {
            'NA': (0.00208*feet).to('meters').magnitude,
            'NB': (0.00208*feet).to('meters').magnitude,
            'NC': (0.00208*feet).to('meters').magnitude,
        }

    @property
    def neutral_strand_resistance(self):
        return {
            'NA': (14.87*ohms / miles).to('ohm / meters').magnitude,
            'NB': (14.87*ohms / miles).to('ohm / meters').magnitude,
            'NC': (14.87*ohms / miles).to('ohm / meters').magnitude,
        }

    @property
    def neutral_strand_diameter(self):
        return {
            'NA': (0.0641*inches).to('meters').magnitude,
            'NB': (0.0641*inches).to('meters').magnitude,
            'NC': (0.0641*inches).to('meters').magnitude,
        }

    @property
    def diameter_over_neutral(self):
        return {
            'NA': (1.29*inches).to('meters').magnitude,
            'NB': (1.29*inches).to('meters').magnitude,
            'NC': (1.29*inches).to('meters').magnitude,
        }

    @property
    def resistance(self):
        return {
            'A': (0.4100*(ohms / miles)).to('ohm / meters').magnitude,
            'B': (0.4100*(ohms / miles)).to('ohm / meters').magnitude,
            'C': (0.4100*(ohms / miles)).to('ohm / meters').magnitude,
        }

    @property
    def geometric_mean_radius(self):
        return {
            'A': (0.0171*feet).to('meters').magnitude,
            'B': (0.0171*feet).to('meters').magnitude,
            'C': (0.0171*feet).to('meters').magnitude,
        }

    @property
    def wire_positions(self):
        return {
            'A': (0, 0),
            'B': ((6*inches).to('meters').magnitude, 0),
            'C': ((12*inches).to('meters').magnitude, 0),
        }

    @property
    def neutral_strand_count(self):
        return {
            'NA': 13,
            'NB': 13,
            'NC': 13,
        }

    @property
    def phases(self):
        return ['A', 'B', 'C', 'NA', 'NB', 'NC']


class IEEE37_Cable(ABCCable):
    """
    Config 723 in IEEE37
    """

    @property
    def diameter_over_neutral(self):
        return {
            'NA': (1.10*inches).to('meters').magnitude,
            'NB': (1.10*inches).to('meters').magnitude,
            'NC': (1.10*inches).to('meters').magnitude,
        }

    @property
    def resistance(self):
        return {
            'A': (0.7690*(ohms / miles)).to('ohm / meters').magnitude,
            'B': (0.7690*(ohms / miles)).to('ohm / meters').magnitude,
            'C': (0.7690*(ohms / miles)).to('ohm / meters').magnitude,
        }

    @property
    def geometric_mean_radius(self):
        return {
            'A': (0.0125*feet).to('meters').magnitude,
            'B': (0.0125*feet).to('meters').magnitude,
            'C': (0.0125*feet).to('meters').magnitude,
        }

    @property
    def neutral_strand_count(self):
        return {
            'NA': 7,
            'NB': 7,
            'NC': 7,
        }


def test_concentric_neutral_cable():
    model = ConcentricNeutralCarsonsEquations(ABCCable())

    assert_array_almost_equal(
        model.impedance,
        array([
            [0.7981 + 1j*0.4467, 0.3188 + 1j*0.0334, 0.2848 + 1j*0.0138],
            [0.3188 + 1j*0.0334, 0.7890 + 1j*0.4048, 0.3188 + 1j*0.0334],
            [0.2848 + 1j*0.0138, 0.3188 + 1j*0.0334, 0.7981 + 1j*0.4467],

        ]) * OHM_PER_MILE_TO_OHM_PER_METER,
        decimal=4
    )


def test_concentric_neutral_cable_IEEE37():
    model = ConcentricNeutralCarsonsEquations(IEEE37_Cable())

    assert_array_almost_equal(
        model.impedance,
        array([
            [1.2936 + 1j*0.6713, 0.4871 + 1j*0.2111, 0.4585 + 1j*0.1521],
            [0.4871 + 1j*0.2111, 1.3022 + 1j*0.6326, 0.4871 + 1j*0.2111],
            [0.4585 + 1j*0.1521, 0.4871 + 1j*0.2111, 1.2936 + 1j*0.6713],

        ]) * OHM_PER_MILE_TO_OHM_PER_METER,
        decimal=4
    )
