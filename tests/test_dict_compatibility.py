from numpy.testing import assert_array_almost_equal

from .test_carsons import ACBN_line_z_primitive
from carsons.carsons import CarsonsEquations


def test_compatibility_with_dict_of_phases():
    class BackwardsCompatibleModel():
        def __init__(self):
            self.resistance = {
                "A": 0.000115575,
                "B": 0.000115575,
                "C": 0.000115575,
                "N": 0.000367852,
            }

            self.geometric_mean_radius = {
                "A": 0.00947938,
                "B": 0.00947938,
                "C": 0.00947938,
                "N": 0.00248107,
            }

            self.wire_positions = {
                "A": (0.762, 8.5344),
                "B": (2.1336, 8.5344),
                "C": (0, 8.5344),
                "N": (1.2192, 7.3152),
            }
            self.phases = {
                "A": "A",
                "B": "B",
                "C": "C",
                "N": "N",
            }
            # we are compatible models that provide 'phases'
            # as a dictionary

    model = BackwardsCompatibleModel()

    z_primative = CarsonsEquations(model).build_z_primitive()
    assert_array_almost_equal(
        z_primative,
        ACBN_line_z_primitive(),
        decimal=4
    )
