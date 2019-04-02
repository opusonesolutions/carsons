import pint

from numpy import array
from numpy.testing import assert_array_almost_equal

from carsons.carsons import ConcentricNeutralCarsonsEquations

from tests.test_carsons import OHM_PER_MILE_TO_OHM_PER_METER

feet = pint.UnitRegistry().feet
inches = pint.UnitRegistry().inches
miles = pint.UnitRegistry().miles
ohms = pint.UnitRegistry().ohms


def GMR_cn(GMR_s, k, R):
    return (GMR_s * k * R**(k-1))**(1/k)


class ABCCable:
    """ Modelled after Kerstings 'Distribution System Modeling and Analysis'
        pg. 102 example """

    @property
    def neutral_strand_geometric_mean_radius(self):
        return {
            'NA': 0.00208*feet.to('meters'),
            'NB': 0.00208*feet.to('meters'),
            'NC': 0.00208*feet.to('meters'),
        }

    def neutral_strand_resistance(self):
        return {
            'NA': (14.87*ohms / miles).to('ohm / meters'),
            'NB': (14.87*ohms / miles).to('ohm / meters'),
            'NC': (14.87*ohms / miles).to('ohm / meters'),
        }

    @property
    def neutral_strand_diameter(self):
        return {
            'NA': 0.0641*inches.to('meters'),
            'NB': 0.0641*inches.to('meters'),
            'NC': 0.0641*inches.to('meters'),
        }

    @property
    def diameter_over_neutral(self):
        return {
            'A': 1.29*inches.to('meters'),
            'B': 1.29*inches.to('meters'),
            'C': 1.29*inches.to('meters'),
        }

    @property
    def resistance(self):
        return {
            'A': (0.4100*ohms / miles).to('ohm / meters'),
            'B': (0.4100*ohms / miles).to('ohm / meters'),
            'C': (0.4100*ohms / miles).to('ohm / meters'),
        }

    @property
    def geometric_mean_radius(self):
        return {
            'A': 0.0171*feet.to('meters'),
            'B': 0.0171*feet.to('meters'),
            'C': 0.0171*feet.to('meters'),
        }

    @property
    def wire_position(self):
        return {
            'A': 0,
            'B': 6*inches.to('meters'),
            'C': 12*inches.to('meters'),
            'NA': 0,
            'NB': 6*inches.to('meters'),
            'NC': 12*inches.to('meters'),
        }


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
