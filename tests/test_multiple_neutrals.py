from carsons.carsons import CarsonsEquations
from tests.helpers import LineModel


def test_dual_neutral_model():

    model = CarsonsEquations(LineModel({
        #    resistance   gmr         (x, y)
        #   ==========================================
        "A":  (0.000115575, 0.00947938, (0.762, 8.5344)),
        "B":  (0.000115575, 0.00947938, (2.1336, 8.5344)),
        "C":  (0.000115575, 0.00947938, (0.0, 8.5344)),
        "N1": (0.000367852, 0.00248107, (0.0, 7.3152)),
        "N2": (0.000367852, 0.00248107, (1.2192, 7.3152)),
    }))
    z_primitive = model.build_z_primitive()
    assert z_primitive.shape == (5, 5)

    # need to get a benchmark case and test the accuracy of the calculation


def test_malformed_neutrals_are_ignored():
    LINE_WITH_BAD_NEUTRAL_LABEL = LineModel({
        #    resistance   gmr         (x, y)
        #   ==========================================
        "A":  (0.000115575, 0.00947938, (0.762, 8.5344)),
        "B":  (0.000115575, 0.00947938, (2.1336, 8.5344)),
        "C":  (0.000115575, 0.00947938, (0.0, 8.5344)),
        "N1": (0.000367852, 0.00248107, (0.0, 7.3152)),
        "pN2": (0.000367852, 0.00248107, (1.2192, 7.3152)),
    })
    model = CarsonsEquations(LINE_WITH_BAD_NEUTRAL_LABEL)

    assert model.build_z_primitive().shape == (4, 4)
