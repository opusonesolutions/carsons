from numpy import array
from numpy.testing import assert_array_almost_equal

from carsons.carsons import TapeShieldedCableCarsonsEquations, calculate_impedance
from tests.test_overhead_line import OHM_PER_MILE_TO_OHM_PER_METER
from tests.test_concentric_neutral_cable import (
    ureg, feet, inches, miles, ohms
)

mils = ureg.mil_length


class AN_Tape_Shielded_Cable():

    def __init__(self, n_pos):
        # parametrize with position of n-phase conductor
        if n_pos:
            self.n_pos = n_pos

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
            'N': ((self.n_pos*inches).to('meters').magnitude, 0),
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


def test_single_phase_tape_shield_with_neutral_Keirsting():
    # Example 4.4, Kiersting, Distribution System Modelling and Analysis
    # 4th Ed, Taylor&Francis, 2018
    # small numerical error found in solution to Kersting example and is adjusted for here

    n_pos = 3 # n phase conductor placed 3 inches away from phase cond

    tape_shielded_cable = TapeShieldedCableCarsonsEquations(AN_Tape_Shielded_Cable(n_pos))
    assert_array_almost_equal(
        calculate_impedance(tape_shielded_cable),
        array([
            [1.3325 + 1j*0.6458, 0.0 + 1j*0.0, 0 + 1j * 0],
            [0 + 1j*0, 0 + 1j*0, 0 + 1j*0],
            [0 + 1j*0, 0 + 1j*0, 0 + 1j*0],

        ]) * OHM_PER_MILE_TO_OHM_PER_METER,
        decimal=5
    )


def test_single_phase_tape_shield_with_neutral_IEEE13():
    # Cable config #607, defined in IEEE13 test feeder
    # https://cmte.ieee.org/pes-testfeeders/resources/
    n_pos = 1 # n phase conductor placed 3 inches away from phase cond

    tape_shielded_cable = TapeShieldedCableCarsonsEquations(AN_Tape_Shielded_Cable(n_pos))
    assert_array_almost_equal(
        calculate_impedance(tape_shielded_cable),
        array([
            [1.3425 + 1j*0.5124, 0.0 + 1j*0.0, 0 + 1j * 0],
            [0 + 1j*0, 0 + 1j*0, 0 + 1j*0],
            [0 + 1j*0, 0 + 1j*0, 0 + 1j*0],

        ]) * OHM_PER_MILE_TO_OHM_PER_METER,
        decimal=5
    )
