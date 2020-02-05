import pint
from numpy import array
from numpy.testing import assert_array_almost_equal

from carsons import MultiConductorCarsonsEquations, calculate_impedance
from tests.helpers import MultiLineModel

ureg = pint.UnitRegistry()

feet = ureg.feet
inches = ureg.inches
miles = ureg.miles
ohms = ureg.ohms
kft = ureg.feet * 1000


def test_triplex_phased_cable():
    """
    Test against triplex NS75 aluminum conductor cable.
    """
    phases = 'ABC'
    phase_conductor = {
        'resistance': (0.611 * (ohms / miles)).to('ohm / meters').magnitude,
        'gmr': (0.014 * feet).to('meters').magnitude,
        'wire_positions': (0, 5),
        'radius': (0.464 / 2 * inches).to('meters').magnitude,
    }
    multi_line_model = MultiLineModel({ph: phase_conductor for ph in phases})
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model), array(
        [[4.388e-4 + 1j * 9.201e-4, 5.922e-5 + 1j * 8.435e-4, 5.922e-5 + 1j * 8.435e-4],
         [5.922e-5 + 1j * 8.435e-4, 4.388e-4 + 1j * 9.201e-4, 5.922e-5 + 1j * 8.435e-4],
         [5.922e-5 + 1j * 8.435e-4, 5.922e-5 + 1j * 8.435e-4, 4.388e-4 + 1j * 9.201e-4]]
    ), decimal=4)
