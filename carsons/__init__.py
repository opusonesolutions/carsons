from carsons.carsons import (  # noqa F401
    CarsonsEquations,
    ConcentricNeutralCarsonsEquations,
    MultiConductorCarsonsEquations,
    TapeShieldedCableCarsonsEquations,
    calculate_impedance,
    calculate_sequence_impedance_matrix,
    calculate_sequence_impedances,
    convert_geometric_model,
)

name = "carsons"


def get_version():
    import sys

    # TODO: if 3.8 support is dropped, we can standardize on
    # importlib.resources.files
    if sys.version_info < (3, 9):
        from importlib.resources import read_text

        return read_text(__name__, "VERSION").strip()
    else:
        from importlib.resources import files

        return files(__name__).joinpath("VERSION").open("r").read().strip()


__version__ = get_version()
