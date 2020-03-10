import pytest
from numpy.testing import assert_array_almost_equal
from numpy import array
from carsons.carsons import (
    CarsonsEquations,
    perform_kron_reduction,
    calculate_sequence_impedance_matrix,
    calculate_sequence_impedances,
)
from tests.test_overhead_line import (
    ACBN_geometry_line,
    CN_geometry_line,
)


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
            'C': (2.1336, 8.5344),
            'B': (0, 8.5344),
            'N': (1.2192, 7.3152),
        }

    @property
    def phases(self):
        return [
            'A',
            'B',
            'C',
            'N',
        ]


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


def ACBN_line_z_primitive():
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


def z_abc_kersting_4_1():
    """Kron-reduced impedance matrix from Kersting 3rd Ed., ex. 4.1"""
    return array([
                [0.4576+1.078j, 0.1560+0.5017j, 0.1535+0.3849j],
                [0.1560+0.5017j, 0.4666+1.*1.0482j, 0.1580+0.4236j],
                [0.1535+0.3849j, 0.1580+0.4236j, 0.4615+1.0651j]])


def z_012_kersting_4_1():
    """Sequence impedance matrix from Kersting 3rd Ed., ex. 4.1"""
    return array([
                [0.7735+1.9373j, 0.0256+0.0115j, -0.0321+0.0159j],
                [-0.0321+0.0159j, 0.3061+0.6270j, -0.0723-0.0060j],
                [0.0256+0.0115j, 0.0723-0.0059j, 0.3061+0.6270j]])


def z_1_kersting_4_1():
    """Positive-sequence impedance from Kersting 3rd Ed., ex. 4.1"""
    return 0.3061 + 0.6270j


def z_0_kersting_4_1():
    """Zero-sequence impedance from Kersting 3rd Ed., ex. 4.1"""
    return 0.7735 + 1.9373j


@pytest.mark.parametrize(
    "line,z_primitive_expected",
    [(ACBN_geometry_line(), ACBN_line_z_primitive()),
     (CN_geometry_line(), CN_line_z_primitive())])
def test_unbalanced_carsons_equations(line, z_primitive_expected):
    model = CarsonsEquations(line)
    z_primitive_computed = model.build_z_primitive()
    assert_array_almost_equal(
        z_primitive_expected,
        z_primitive_computed,
    )


@pytest.mark.parametrize(
    "line,z_primitive_expected",
    [(ABCN_balanced_line(), ABCN_balanced_z_primitive())])
def test_balanced_carsons_equations(line, z_primitive_expected):
    model = CarsonsEquations(line)
    z_primitive_computed = model.build_z_primitive()
    assert_array_almost_equal(
        z_primitive_expected,
        z_primitive_computed,
    )


@pytest.mark.parametrize(
    "z_primitive,expected_z_abc",
    [(z_primitive_no_neutral(), z_primitive_no_neutral()),
     (z_primitive_one_neutral(), expected_z_abc_one_neutral()),
     (z_primitive_three_neutrals(), expected_z_abc_three_neutrals())])
def test_kron_reduction(z_primitive, expected_z_abc):
    actual_z_abc = perform_kron_reduction(z_primitive)
    assert (actual_z_abc == expected_z_abc).all()


@pytest.mark.parametrize(
    "z_abc,z_012_expected",
    [(z_abc_kersting_4_1(), z_012_kersting_4_1())])
def test_sequence_impedance_matrix(z_abc, z_012_expected):
    z_012_computed = calculate_sequence_impedance_matrix(z_abc)
    assert_array_almost_equal(z_012_expected, z_012_computed, decimal=0.001)


@pytest.mark.parametrize(
    "z_abc,expected_z1,expected_z0",
    [(z_abc_kersting_4_1(), z_1_kersting_4_1(), z_0_kersting_4_1())])
def test_sequence_impedance(z_abc, expected_z1, expected_z0):
    actual_z1, actual_z0 = calculate_sequence_impedances(z_abc)
    assert actual_z1.real == pytest.approx(expected_z1.real, 0.001)
    assert actual_z1.imag == pytest.approx(expected_z1.imag, 0.001)
    assert actual_z0.real == pytest.approx(expected_z0.real, 0.001)
    assert actual_z0.imag == pytest.approx(expected_z0.imag, 0.001)
