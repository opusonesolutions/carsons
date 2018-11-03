import pytest

from carsons.phase_model import A, B, C, N, Neutral


@pytest.mark.parametrize('object, other_element', [
    (A, "A"),
    (B, "B"),
    (C, "C"),
    (N, "N"),
    (Neutral("N1"), "N1")
])
def test_equality(object, other_element):
    assert object == other_element
    assert other_element == object


def test_custom_neutral():
    assert isinstance(N, Neutral)
    assert Neutral("N1") is not Neutral("N2")
    assert Neutral("N1") is not N


def test_neutrals_must_start_with_n():
    with pytest.raises(ValueError) as e:
        Neutral("pn1")

    assert "Neutral label must start with 'N'; got 'pn1'" in str(e)


def test_sorting_phase_objects():
    result = sorted([B, N, C, Neutral('N1'), A])

    assert result == [A, B, C, N, Neutral('N1')]
