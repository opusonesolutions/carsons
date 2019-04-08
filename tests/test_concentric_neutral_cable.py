import pint
from numpy import array
from numpy.testing import assert_array_almost_equal

from carsons.carsons import ConcentricNeutralCarsonsEquations
from tests.helpers import ConcentricLineModel
from tests.test_carsons import OHM_PER_MILE_TO_OHM_PER_METER

ureg = pint.UnitRegistry()

feet = ureg.feet
inches = ureg.inches
miles = ureg.miles
ohms = ureg.ohms


def test_concentric_neutral_cable():
    model = ConcentricNeutralCarsonsEquations(ConcentricLineModel({
        "A": {
            'resistance': (0.4100*(ohms / miles)).to('ohm / meters').magnitude,
            'gmr': (0.0171*feet).to('meters').magnitude,
            'wire_positions': (0, 0)
        },
        "B": {
            'resistance': (0.4100*(ohms / miles)).to('ohm / meters').magnitude,
            'gmr': (0.0171*feet).to('meters').magnitude,
            'wire_positions': ((6*inches).to('meters').magnitude, 0)
        },
        "C": {
            'resistance': (0.4100*(ohms / miles)).to('ohm / meters').magnitude,
            'gmr': (0.0171*feet).to('meters').magnitude,
            'wire_positions': ((12*inches).to('meters').magnitude, 0)
        },

        "NA": {
            'neutral_strand_gmr': (0.00208*feet).to('meters').magnitude,
            'neutral_strand_resistance':
                (14.87*ohms / miles).to('ohm / meters').magnitude,
            'neutral_strand_diameter': (0.0641*inches).to('meters').magnitude,
            'diameter_over_neutral': (1.29*inches).to('meters').magnitude,
            'neutral_strand_count': 13,
        },
        "NB": {
            'neutral_strand_gmr': (0.00208*feet).to('meters').magnitude,
            'neutral_strand_resistance':
                (14.87*ohms / miles).to('ohm / meters').magnitude,
            'neutral_strand_diameter': (0.0641*inches).to('meters').magnitude,
            'diameter_over_neutral': (1.29*inches).to('meters').magnitude,
            'neutral_strand_count': 13,
        },
        "NC": {
            'neutral_strand_gmr': (0.00208*feet).to('meters').magnitude,
            'neutral_strand_resistance':
                (14.87*ohms / miles).to('ohm / meters').magnitude,
            'neutral_strand_diameter': (0.0641*inches).to('meters').magnitude,
            'diameter_over_neutral': (1.29*inches).to('meters').magnitude,
            'neutral_strand_count': 13,
        },
    }))

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
    model = ConcentricNeutralCarsonsEquations(ConcentricLineModel({
        "A": {
            'resistance': (0.7690 * (ohms/miles)).to('ohm / meters').magnitude,
            'gmr': (0.0125 * feet).to('meters').magnitude,
            'wire_positions': (0, 0)},
        "B": {
            'resistance': (0.7690 * (ohms/miles)).to('ohm / meters').magnitude,
            'gmr': (0.0125 * feet).to('meters').magnitude,
            'wire_positions': ((6 * inches).to('meters').magnitude, 0)},
        "C": {
            'resistance': (0.7690 * (ohms/miles)).to('ohm / meters').magnitude,
            'gmr': (0.0125 * feet).to('meters').magnitude,
            'wire_positions': ((12 * inches).to('meters').magnitude, 0)},

        "NA": {
            'neutral_strand_gmr': (0.00208 * feet).to('meters').magnitude,
            'neutral_strand_resistance':
                (14.87 * ohms / miles).to('ohm / meters').magnitude,
            'neutral_strand_diameter': (0.0641*inches).to('meters').magnitude,
            'diameter_over_neutral': (1.10 * inches).to('meters').magnitude,
            'neutral_strand_count': 7, },
        "NB": {
            'neutral_strand_gmr': (0.00208 * feet).to('meters').magnitude,
            'neutral_strand_resistance':
                (14.87 * ohms / miles).to('ohm / meters').magnitude,
            'neutral_strand_diameter': (0.0641*inches).to('meters').magnitude,
            'diameter_over_neutral': (1.10 * inches).to('meters').magnitude,
            'neutral_strand_count': 7, },
        "NC": {
            'neutral_strand_gmr': (0.00208 * feet).to('meters').magnitude,
            'neutral_strand_resistance':
                (14.87 * ohms / miles).to('ohm / meters').magnitude,
            'neutral_strand_diameter': (0.0641*inches).to('meters').magnitude,
            'diameter_over_neutral': (1.10 * inches).to('meters').magnitude,
            'neutral_strand_count': 7, }, }))

    assert_array_almost_equal(
        model.impedance,
        array([
            [1.2936 + 1j*0.6713, 0.4871 + 1j*0.2111, 0.4585 + 1j*0.1521],
            [0.4871 + 1j*0.2111, 1.3022 + 1j*0.6326, 0.4871 + 1j*0.2111],
            [0.4585 + 1j*0.1521, 0.4871 + 1j*0.2111, 1.2936 + 1j*0.6713],

        ]) * OHM_PER_MILE_TO_OHM_PER_METER,
        decimal=4
    )
