from numpy import array
from numpy.testing import assert_array_almost_equal

from carsons.carsons import TapeShieldedCable, calculate_impedance
from tests.test_overhead_line import OHM_PER_MILE_TO_OHM_PER_METER
from tests.test_concentric_neutral_cable import (
    ureg, feet, inches, miles, ohms
)

mils = ureg.mils


class AN_Tape_Shielded_Cable():

    @property
    def resistance(self):
        return {
            'A': (0.97*(ohms / miles)).to('ohm / meters').magnitude,
            'N': (0.607*(ohms / miles)).to('ohm / meters').magnitude,
        }

    @property
    def geometric_mean_radius(self):
        return {
            'A': (0.0111*feet).to('meters').magnitude,
            'N': (0.01113*feet).to('meters').magnitude,
        }

    @property
    def wire_positions(self):
        return {
            'A': (0, 0),
            'N': ((3*inches).to('meters').magnitude, 0),
        }

    @property
    def tape_shield_outer_diameter(self):
        return {
            'A': (0.88*inches).to('meters').magnitude,
        }

    @property
    def tape_shield_thickness(self):
        return {
            'A': (5*mils).to('meters').magnitude,
        }

    @property
    def phases(self):
        return [
            'A',
            'N',
        ]


def test_single_phase_tape_shield_with_neutral():

    tape_shielded_cable = TapeShieldedCable(AN_Tape_Shielded_Cable())
    assert_array_almost_equal(
        calculate_impedance(tape_shielded_cable),
        array([
            [1.3218 + 1j*0.6744, 0.0 + 1j*0.0, 0 + 1j * 0],
            [0 + 1j*0, 0 + 1j*0, 0 + 1j*0],
            [0 + 1j*0, 0 + 1j*0, 0 + 1j*0],

        ]) * OHM_PER_MILE_TO_OHM_PER_METER,
        decimal=5
    )
