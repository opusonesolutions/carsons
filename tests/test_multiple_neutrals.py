import numpy
from numpy.testing import assert_array_almost_equal

from carsons.carsons import CarsonsEquations
from tests.helpers import LineModel
from tests.test_carsons import ACBN_line_z_primitive


def test_dual_neutral_model():

    model = CarsonsEquations(LineModel({
        #    resistance   gmr         (x, y)
        #   ==========================================
        "A":  (0.000115575, 0.00947938, (0.762, 8.5344)),
        "B":  (0.000115575, 0.00947938, (0.0, 8.5344)),
        "N1":  (0.000115575, 0.00947938, (2.1336, 8.5344)),
        "N2": (0.000367852, 0.00248107, (1.2192, 7.3152)),
    }))
    z_primitive = model.build_z_primitive()

    # dimensions should include A, B, C and as many neutrals as are described
    assert z_primitive.shape == (5, 5)

    # because there's no C conductor in the model, expect a row/column of zeros
    assert_array_almost_equal(z_primitive[2, 0:], [0+0j]*5)
    assert_array_almost_equal(z_primitive[0:, 2], [0+0j]*5)

    # if we delete the C row/column, we should get a z_primitive that looks
    # like Configuration 601, since the geometry of A/B/C/N is mapped to
    # A/B/N1/N2
    z_equivalent = z_primitive.copy()
    z_equivalent = numpy.delete(z_equivalent, 2, 0)
    z_equivalent = numpy.delete(z_equivalent, 2, 1)
    z_expected = ACBN_line_z_primitive()
    assert_array_almost_equal(z_equivalent, z_expected)


def test_malformed_neutrals_are_ignored():
    LINE_WITH_BAD_NEUTRAL_LABEL = LineModel({
        #    resistance   gmr         (x, y)
        #   ==========================================
        "A":  (0.000115575, 0.00947938, (0.762, 8.5344)),
        "C":  (0.000115575, 0.00947938, (2.1336, 8.5344)),
        "B":  (0.000115575, 0.00947938, (0.0, 8.5344)),
        "N1": (0.000367852, 0.00248107, (1.2192, 7.3152)),
        "pN2": (0.000367852, 0.00248107, (0.0, 7.3152)),
    })
    model = CarsonsEquations(LINE_WITH_BAD_NEUTRAL_LABEL)
    z_primitive = model.build_z_primitive()

    assert z_primitive.shape == (4, 4)

    z_expected = ACBN_line_z_primitive()
    assert_array_almost_equal(z_primitive, z_expected)
