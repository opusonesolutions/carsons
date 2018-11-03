import pytest
from numpy.testing import assert_array_almost_equal
from numpy import array
from carsons.carsons import (
    CarsonsEquations,
    convert_geometric_model,
    perform_kron_reduction,
)

# `carsons` implements the model entirely in SI metric units, however this
# conversion allows us to enter in impedance as ohm-per-mile in the test
# harness, which means we can lift matrices directly out of the ieee4 test
# network.
OHM_PER_MILE_TO_OHM_PER_METER = 1 / 1609.344


class ABCN_geometry_line():

    @property
    def resistance(self):
        return {
            'A': 0.000115575,
            'B': 0.000115575,
            'C': 0.000115575,
            'N': 0.000367852,
        }

    @property
    def geometric_mean_radius(self):
        return {
            'A': 0.00947938,
            'B': 0.00947938,
            'C': 0.00947938,
            'N': 0.00248107,
        }

    @property
    def wire_positions(self):
        return {
            'A': (0.762, 8.5344),
            'B': (2.1336, 8.5344),
            'C': (0, 8.5344),
            'N': (1.2192, 7.3152),
        }

    @property
    def phases(self):
        return {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'N': 'N',
        }


class CBN_geometry_line():

    @property
    def resistance(self):
        return {
            'B': 0.000695936,
            'C': 0.000695936,
            'N': 0.000695936,
        }

    @property
    def geometric_mean_radius(self):
        return {
            'B': 0.00135941,
            'C': 0.00135941,
            'N': 0.00135941,
        }

    @property
    def wire_positions(self):
        return {
            'B': (2.1336, 8.5344),
            'C': (0, 8.5344),
            'N': (1.2192, 7.3152),
        }

    @property
    def phases(self):
        return {
            'B': 'B',
            'C': 'C',
            'N': 'N',
        }


class CN_geometry_line():

    @property
    def resistance(self):
        return {
            'C': 0.000695936,
            'N': 0.000695936,
        }

    @property
    def geometric_mean_radius(self):
        return {
            'C': 0.00135941,
            'N': 0.00135941,
        }

    @property
    def wire_positions(self):
        return {
            'C': (0, 8.8392),
            'N': (0.1524, 7.3152),
        }

    @property
    def phases(self):
        return {
            'C': 'C',
            'N': 'N',
        }


class ABCN_balanced_line():

    @property
    def resistance(self):
        return {
            'A': 0.000115575,
            'B': 0.000115575,
            'C': 0.000115575,
            'N': 0.000115575,
        }

    @property
    def geometric_mean_radius(self):
        return {
            'A': 0.00947938,
            'B': 0.00947938,
            'C': 0.00947938,
            'N': 0.00947938,
        }

    @property
    def wire_positions(self):
        return {
            'A': (0.762, 8.5344),
            'B': (2.1336, 8.5344),
            'C': (0, 8.5344),
            'N': (1.2192, 7.3152),
        }

    @property
    def phases(self):
        return {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'N': 'N',
        }


def ABCN_line_geometry_phase_impedance():
    return OHM_PER_MILE_TO_OHM_PER_METER * array([
            [0.3465 + 1.0179j, 0.1560 + 0.5017j, 0.1580 + 0.4236j],
            [0.1560 + 0.5017j, 0.3375 + 1.0478j, 0.1535 + 0.3849j],
            [0.1580 + 0.4236j, 0.1535 + 0.3849j, 0.3414 + 1.0348j]])


def ABCN_line_z_primitive():
    return array([
        [1.74792626e-04+0.00085989j,
         5.92176264e-05+0.00052913j,
         5.92176264e-05+0.00048481j,
         5.92176264e-05+0.00048873j],
        [5.92176264e-05+0.00052913j,
         1.74792626e-04+0.00085989j,
         5.92176264e-05+0.0004515j,
         5.92176264e-05+0.00046756j],
        [5.92176264e-05+0.00048481j,
         5.92176264e-05+0.0004515j,
         1.74792626e-04+0.00085989j,
         5.92176264e-05+0.00047687j],
        [5.92176264e-05+0.00048873j,
         5.92176264e-05+0.00046756j,
         5.92176264e-05+0.00047687j,
         4.27069626e-04+0.00096095j]])


def ABCN_balanced_z_primitive():
    return array([
        [1.74792626e-04+0.00085989j,
         5.92176264e-05+0.00052913j,
         5.92176264e-05+0.00048481j,
         5.92176264e-05+0.00048873j],
        [5.92176264e-05+0.00052913j,
         1.74792626e-04+0.00085989j,
         5.92176264e-05+0.0004515j,
         5.92176264e-05+0.00046756j],
        [5.92176264e-05+0.00048481j,
         5.92176264e-05+0.0004515j,
         1.74792626e-04+0.00085989j,
         5.92176264e-05+0.00047687j],
        [5.92176264e-05+0.00048873j,
         5.92176264e-05+0.00046756j,
         5.92176264e-05+0.00047687j,
         1.74792626e-04+0.00085989j]
    ])


def CBN_line_geometry_phase_impedance():
    return OHM_PER_MILE_TO_OHM_PER_METER * array([
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, 1.3294 + 1.3471j, 0.2066 + 0.4591j],
            [0.0000 + 0.0000j, 0.2066 + 0.4591j, 1.3238 + 1.3569j]])


def CN_line_geometry_phase_impedance():
    return OHM_PER_MILE_TO_OHM_PER_METER * array([
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, 0.0000 + 0.0000j],
            [0.0000 + 0.0000j, 0.0000 + 0.0000j, 1.3292 + 1.3475j]])


def CN_line_z_primitive():
    return array([
            [0.0+0.j, 0.0+0.j, 0.0+0.j, 0.0+0.j],
            [0.0+0.j, 0.0+0.j, 0.0+0.j, 0.0+0.j],
            [0.0+0.j, 0.0+0.j, 7.5515e-04+1.006e-3j, 5.9217e-05+0.00047649j],
            [0.0+0.j, 0.0+0.j, 5.921e-05+0.00047649j, 7.5515e-04+0.00100631j]])


def z_primitive_no_neutral():
    return array([[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                  [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                  [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]])


def z_primitive_one_neutral():
    return array([[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                  [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                  [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
                  [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]])


def expected_z_abc_one_neutral():
    return array([
                [0.0 + 0.0j, -1.0 + 0.0j, -1.0 + 0.0j],
                [-1.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
                [-1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j]])


def z_primitive_three_neutrals():
    return array([
            [1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 2 + 0j, 3 + 0j],
            [0 + 0j, 1 + 0j, 0 + 0j, 1 + 0j, 2 + 0j, 3 + 0j],
            [0 + 0j, 0 + 0j, 1 + 0j, 1 + 0j, 2 + 0j, 3 + 0j],
            [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
            [2 + 0j, 2 + 0j, 2 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
            [3 + 0j, 3 + 0j, 3 + 0j, 0 + 0j, 0 + 0j, 1 + 0j]])


def expected_z_abc_three_neutrals():
    return array([
                [-13 + 0j, -14 + 0j, -14 + 0j],
                [-14 + 0j, -13 + 0j, -14 + 0j],
                [-14 + 0j, -14 + 0j, -13 + 0j]])


@pytest.mark.parametrize(
    "line,expected_impedance",
    [(ABCN_geometry_line(), ABCN_line_geometry_phase_impedance()),
     (CBN_geometry_line(), CBN_line_geometry_phase_impedance()),
     (CN_geometry_line(), CN_line_geometry_phase_impedance())])
def test_converts_geometry_to_phase_impedance(line, expected_impedance):
    actual_impedance = convert_geometric_model(line)
    assert_array_almost_equal(expected_impedance,
                              actual_impedance,
                              decimal=4)


@pytest.mark.parametrize(
    "line,z_primitive_expected",
    [(ABCN_geometry_line(), ABCN_line_z_primitive()),
     (CN_geometry_line(), CN_line_z_primitive())])
def test_unbalanced_carsons_equations(line, z_primitive_expected):
    model = CarsonsEquations(line)
    z_primitive_computed = model.build_z_primitive()
    assert_array_almost_equal(
        z_primitive_expected,
        z_primitive_computed,
        decimal=4)


@pytest.mark.parametrize(
    "line,z_primitive_expected",
    [(ABCN_balanced_line(), ABCN_balanced_z_primitive())])
def test_balanced_carsons_equations(line, z_primitive_expected):
    model = CarsonsEquations(line)
    z_primitive_computed = model.build_z_primitive()
    assert_array_almost_equal(
        z_primitive_expected,
        z_primitive_computed,
        decimal=4
    )


@pytest.mark.parametrize(
    "z_primitive,expected_z_abc",
    [(z_primitive_no_neutral(), z_primitive_no_neutral()),
     (z_primitive_one_neutral(), expected_z_abc_one_neutral()),
     (z_primitive_three_neutrals(), expected_z_abc_three_neutrals())])
def test_kron_reduction(z_primitive, expected_z_abc):
    actual_z_abc = perform_kron_reduction(z_primitive)
    assert (actual_z_abc == expected_z_abc).all()


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

    model = BackwardsCompatibleModel()

    z_primative = CarsonsEquations(model).build_z_primitive()
    assert_array_almost_equal(
        z_primative,
        ABCN_line_z_primitive(),
        decimal=4
    )
