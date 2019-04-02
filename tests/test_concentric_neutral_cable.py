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


def GMR_cn(GMR_s, k, R):
    return (GMR_s * k * R**(k-1))**(1/k)


class ABCCable:
    """ Modelled after Kerstings 'Distribution System Modeling and Analysis'
        pg. 102 example """

    @property
    def neutral_strand_geometric_mean_radius(self):
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
            'NA': (0, 0),
            'NB': ((6*inches).to('meters').magnitude, 0),
            'NC': ((12*inches).to('meters').magnitude, 0),
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


def test_concentric_neutral_cable():
    model = ConcentricNeutralCarsonsEquations(ABCCable())

    assert_array_almost_equal(
        model.impedance,
        array([
            [0.7981 + 1j*0.4467, 0.3188 + 1j*0.0334, 0.2848 + 1j*0.0138],
            [0.3188 + 1j*0.0334, 0.7890 + 1j*0.4048, 0.3188 + 1j*0.0334],
            [0.2848 + 1j*0.0138, 0.3188 + 1j*0.0334, 0.7981 + 1j*0.4467],

        ]) * OHM_PER_MILE_TO_OHM_PER_METER
    )
