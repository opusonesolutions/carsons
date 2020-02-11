import pint
import pytest
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
        {'S1': phase_conductor, 'S2': phase_conductor, 'N': neutral_conductor}
    )
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model), array(
        [[1.5304 + 1j * 0.6132, 0.5574 + 1j * 0.4461],
         [0.5574 + 1j * 0.4461, 1.5304 + 1j * 0.6132]]
    ) * OHM_PER_MILE_TO_OHM_PER_METER, decimal=4)


EXPECTED_DUPLEX_IMPEDANCE = array(
        [[0.0 + 1j*0.0, 0.0 + 1j*0.0, 0.0 + 1j*0.0],
         [0.0 + 1j*0.0, 9.521e-4 + 1j*3.744e-4, 0.0 + 1j*0.0],
         [0.0 + 1j*0.0, 0.0 + 1j*0.0, 0.0 + 1j*0.0]]
    )

EXPECTED_QUADRUPLEX_IMPEDANCE = array(
        [[5.223e-4 + 1j*2.279e-4, 2.216e-4 + 1j*1.373e-4,
          2.216e-4 + 1j*1.373e-4],
         [2.216e-4 + 1j*1.373e-4, 5.223e-4 + 1j*2.279e-4,
          2.216e-4 + 1j*1.373e-4],
         [2.216e-4 + 1j*1.373e-4, 2.216e-4 + 1j*1.373e-4,
          5.223e-4 + 1j*2.279e-4]]
    )


@pytest.mark.parametrize(
    'phases, resistance, gmr, wire_position, radius, insulation_thickness, '
    'expected_result',
    [('B', 0.97, 0.0111, (0, 5), 0.368/2, 0.00137, EXPECTED_DUPLEX_IMPEDANCE),
     ('ABC', 0.484, 0.0158, (0, 5), 0.522/2, 0.00137,
      EXPECTED_QUADRUPLEX_IMPEDANCE)]
)
def test_multi_conductor_cable_with_neutral(
        phases, resistance, gmr, wire_position, radius,
        insulation_thickness, expected_result
):
    conductor = {
        'resistance': (resistance * (ohms / miles)
                       ).to('ohm / meters').magnitude,
        'gmr': (gmr * feet).to('meters').magnitude,
        'wire_positions': wire_position,
        'radius': (radius * inches).to('meters').magnitude,
    }
    phase_conductor = {**conductor,
                       'insulation_thickness': insulation_thickness}
    neutral_conductor = {**conductor, 'insulation_thickness': 0}

    line_model_dict = {ph: phase_conductor for ph in phases}
    line_model_dict.update({'N': neutral_conductor})

    multi_line_model = MultiLineModel(line_model_dict)
    carsons_model = MultiConductorCarsonsEquations(multi_line_model)

    assert_array_almost_equal(calculate_impedance(carsons_model),
                              expected_result, decimal=4)
