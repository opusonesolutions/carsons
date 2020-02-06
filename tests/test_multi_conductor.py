import pint
from numpy import array
from numpy.testing import assert_array_almost_equal

from carsons import MultiConductorCarsonsEquations, calculate_impedance
from tests.helpers import MultiLineModel
from tests.test_overhead_line import OHM_PER_MILE_TO_OHM_PER_METER

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


def test_triplex_secondary():
    """
    Test against 1/0 AA triplex from example 11.3 in Kersting's book.
    """
    conductor = {
        'resistance': (0.97 * (ohms / miles)).to('ohm / meters').magnitude,
        'gmr': (0.0111 * feet).to('meters').magnitude,
        'wire_positions': (0, 1),
        'radius': (0.368 / 2 * inches).to('meters').magnitude,
    }
    phase_conductor = {**conductor, 'insulation_thickness': (0.08 * inches).to('meters').magnitude}
    neutral_conductor = {**conductor, 'insulation_thickness': 0}
    multi_line_model = MultiLineModel(
        {'A': phase_conductor, 'N': neutral_conductor}, is_secondary=True
    )
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model), array(
        [[1.5304 + 1j * 0.6132, 0.5574 + 1j * 0.4461],
         [0.5574 + 1j * 0.4461, 1.5304 + 1j * 0.6132]]
    ) * OHM_PER_MILE_TO_OHM_PER_METER, decimal=4)
