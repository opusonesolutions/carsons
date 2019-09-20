from itertools import combinations_with_replacement
from math import sqrt
import numpy
from numpy.testing import assert_array_almost_equal
import pytest

from carsons.carsons import CarsonsEquations
from tests.helpers import LineModel


def test_impedances_with_50hz_frequency():
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

    # this test rests on the assumption that the real parts of the off-diagonal
    # elements in z_primitive are proportional to frequency:
    #  where i ≠ j, R == μω / π Pᵢⱼ
    #
    # This holds for P approximated to just one term; additional terms
    # incorporate kᵢⱼ, and kᵢⱼ ∝ √ω

    antidiagonal_mask = 1 - numpy.identity(4)

    assert_array_almost_equal(
        z_primitive_50hz.real * antidiagonal_mask,
        z_primitive_60hz.real * antidiagonal_mask * 50/60
    )


@pytest.mark.parametrize('i, j', combinations_with_replacement("ABCN", 2))
def test_k_for_50_vs_60hz_models(i, j):
    line_model = LineModel({
        #     resistance   gmr         (x, y)
        #   ==========================================
        "A": (0.000115575, 0.00947938, (0.762, 8.5344)),
        "B": (0.000115575, 0.00947938, (0.0, 8.5344)),
        "C": (0.000115575, 0.00947938, (2.1336, 8.5344)),
        "N": (0.000367852, 0.00248107, (1.2192, 7.3152)),
    })
    model_60hz = CarsonsEquations(line_model)

    line_model.frequency = 50
    model_50hz = CarsonsEquations(line_model)

    # k is proportional to √ω
    expected_ratio = sqrt(50) / sqrt(60)
    k_50hz = model_50hz.compute_k(i, j)
    k_60hz = model_60hz.compute_k(i, j)

    assert k_50hz == pytest.approx(k_60hz * expected_ratio)
