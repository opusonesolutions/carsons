import numpy
from numpy.testing import assert_array_almost_equal

from carsons.carsons import CarsonsEquations
from tests.helpers import LineModel


def test_dual_neutral_model():
    line_model = LineModel({
        #     resistance   gmr         (x, y)
        #   ==========================================
        "A": (0.000115575, 0.00947938, (0.762, 8.5344)),
        "B": (0.000115575, 0.00947938, (0.0, 8.5344)),
        "C": (0.000115575, 0.00947938, (2.1336, 8.5344)),
        "N": (0.000367852, 0.00248107, (1.2192, 7.3152)),
    })
    model = CarsonsEquations(line_model)

    z_primitive_60hz = model.build_z_primitive()
    line_model.frequency = 50
    z_primitive_50hz = CarsonsEquations(line_model).build_z_primitive()

    antidiagonal_mask = 1 - numpy.identity(4)

    assert_array_almost_equal(
        z_primitive_50hz * antidiagonal_mask,
        z_primitive_60hz * antidiagonal_mask * 50/60
    )
