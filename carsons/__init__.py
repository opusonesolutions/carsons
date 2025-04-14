from carsons.carsons import (
    CarsonsEquations,
    ConcentricNeutralCarsonsEquations,
    MultiConductorCarsonsEquations,
    TapeShieldedCableCarsonsEquations,
    calculate_impedance,
    calculate_sequence_impedance_matrix,
    calculate_sequence_impedances,
    convert_geometric_model,
)

__all__ = [
    "CarsonsEquations",
    "ConcentricNeutralCarsonsEquations",
    "MultiConductorCarsonsEquations",
    "TapeShieldedCableCarsonsEquations",
    "calculate_impedance",
    "calculate_sequence_impedance_matrix",
    "calculate_sequence_impedances",
    "convert_geometric_model",
]

name = "carsons"


def get_version():
    from importlib.resources import files

    return files(__name__).joinpath("VERSION").open("r").read().strip()


__version__ = get_version()
