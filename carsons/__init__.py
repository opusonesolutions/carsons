from carsons.carsons import (convert_geometric_model,               # noqa 401
                             calculate_impedance,                   # noqa 401
                             calculate_sequence_impedance_matrix,
                             calculate_sequence_impedances,
                             CarsonsEquations,
                             ConcentricNeutralCarsonsEquations,     # noqa 401
                             MultiConductorCarsonsEquations,        # noqa 401
                             TapeShieldedCableCarsonsEquations)        # noqa 401

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
