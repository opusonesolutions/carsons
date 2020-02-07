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


def test_duplex_1ph():
    """
    Test 1/0 NS75 duplex cable.
    """
    phases = 'B'
    conductor = {
        'resistance': (0.97 * (ohms / miles)).to('ohm / meters').magnitude,
        'gmr': (0.0111 * feet).to('meters').magnitude,
        'wire_positions': (0, 5),
        'radius': (0.368 / 2 * inches).to('meters').magnitude,
    }
    phase_conductor = {**conductor, 'insulation_thickness': 0.00137}
    neutral_conductor = {**conductor, 'insulation_thickness': 0}

    line_model_dict = {ph: phase_conductor for ph in phases}
    line_model_dict.update({'N': neutral_conductor})

    multi_line_model = MultiLineModel(line_model_dict)
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model), array(
        [[0.0 + 1j*0.0, 0.0 + 1j*0.0, 0.0 + 1j*0.0],
         [0.0 + 1j*0.0, 9.521e-4 + 1j*3.744e-4, 0.0 + 1j*0.0],
         [0.0 + 1j*0.0, 0.0 + 1j*0.0, 0.0 + 1j*0.0]]
    ), decimal=4)


def test_triplex_phased_cable():
    """
    Test against 3/0 triplex NS75 aluminum conductor cable.
    """
    phases = 'ABC'
    phase_conductor = {
        'resistance': (0.611 * (ohms / miles)).to('ohm / meters').magnitude,
        'gmr': (0.014 * feet).to('meters').magnitude,
        'wire_positions': (0, 5),
        'radius': (0.464 / 2 * inches).to('meters').magnitude,
        'insulation_thickness': 0.00137,
    }
    multi_line_model = MultiLineModel({ph: phase_conductor for ph in phases})
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model), array(
        [[4.389e-4 + 1j*9.200e-4, 5.922e-5 + 1j*8.277e-4,
          5.922e-5 + 1j*8.277e-4],
         [5.922e-5 + 1j*8.277e-4, 4.389e-4 + 1j*9.200e-4,
          5.922e-5 + 1j*8.277e-4],
         [5.922e-5 + 1j*8.277e-4, 5.922e-5 + 1j*8.277e-4,
          4.389e-4 + 1j*9.200e-4]]
    ), decimal=4)


def test_triplex_secondary():
    """
    Test against 1/0 AA triplex secondary cable from example 11.3 on page 463
    of 'Distribution System Modeling and Analysis' by William H.Kersting,
    4th edition.
    """
    conductor = {
        'resistance': (0.97 * (ohms / miles)).to('ohm / meters').magnitude,
        'gmr': (0.0111 * feet).to('meters').magnitude,
        'wire_positions': (0, 1),
        'radius': (0.368 / 2 * inches).to('meters').magnitude,
    }
    phase_conductor = {
        **conductor,
        'insulation_thickness': (0.08 * inches).to('meters').magnitude
    }
    neutral_conductor = {**conductor, 'insulation_thickness': 0}
    multi_line_model = MultiLineModel(
        {'A': phase_conductor, 'N': neutral_conductor}, is_secondary=True
    )
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model), array(
        [[1.5304 + 1j * 0.6132, 0.5574 + 1j * 0.4461],
         [0.5574 + 1j * 0.4461, 1.5304 + 1j * 0.6132]]
    ) * OHM_PER_MILE_TO_OHM_PER_METER, decimal=4)


def test_quadruplex_3ph():
    """
    Test 4/0 NS75 quadruplex cable.
    """
    phases = 'ABC'
    conductor = {
        'resistance': (0.484 * (ohms / miles)).to('ohm / meters').magnitude,
        'gmr': (0.0158 * feet).to('meters').magnitude,
        'wire_positions': (0, 5),
        'radius': (0.522 / 2 * inches).to('meters').magnitude,
    }
    phase_conductor = {**conductor, 'insulation_thickness': 0.00137}
    neutral_conductor = {**conductor, 'insulation_thickness': 0}

    line_model_dict = {ph: phase_conductor for ph in phases}
    line_model_dict.update({'N': neutral_conductor})

    multi_line_model = MultiLineModel(line_model_dict)
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model), array(
        [[5.223e-4 + 1j*2.279e-4, 2.216e-4 + 1j*1.373e-4,
          2.216e-4 + 1j*1.373e-4],
         [2.216e-4 + 1j*1.373e-4, 5.223e-4 + 1j*2.279e-4,
          2.216e-4 + 1j*1.373e-4],
         [2.216e-4 + 1j*1.373e-4, 2.216e-4 + 1j*1.373e-4,
          5.223e-4 + 1j*2.279e-4]]
    ), decimal=4)
