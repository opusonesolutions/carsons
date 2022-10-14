from collections import defaultdict
from itertools import islice
from typing import Dict, Iterable, Iterator, Tuple

from numpy import arctan, cos, log, sin, sqrt, zeros, exp
from numpy import array, ndarray
from numpy import pi as π
from numpy.linalg import inv

alpha = exp(2j*π/3)

A = array([
            [1, 1, 1],
            [1, alpha**2, alpha],
            [1, alpha, alpha**2],
])

Ainv = (1/3)*array([
                [1, 1, 1],
                [1, alpha, alpha**2],
                [1, alpha**2, alpha],
])


def convert_geometric_model(geometric_model) -> ndarray:
    carsons_model = CarsonsEquations(geometric_model)

    z_primitive = carsons_model.build_z_primitive()
    z_abc = perform_kron_reduction(z_primitive)
    return z_abc


def calculate_impedance(model) -> ndarray:
    z_primitive = model.build_z_primitive()
    z_abc = perform_kron_reduction(z_primitive, dimension=model.dimension)

    return z_abc


def perform_kron_reduction(z_primitive: ndarray, dimension=3) -> ndarray:
    """ Reduces the primitive impedance matrix to an equivalent impedance
        matrix.

        We break z_primative up into four quadrants as follows:

              Ẑpp = [Ẑaa, Ẑab, Ẑac]   Ẑpn = [Ẑan]
                    [Ẑba, Ẑbb, Ẑbc]         [Ẑbn]
                    [Ẑca, Ẑcb, Ẑcc]         [Ẑcn]

              Ẑnp = [Ẑna, Ẑnb, Ẑnc]   Ẑnn = [Ẑnn]

        Ẑnn is of dimension mxm, where m is the number of neutrals. E.g. with
        m = 2:
                                          Ẑan = [Ẑan₁,  Ẑan₂]
                                                [Ẑbn₁,  Ẑbn₂]
                                                [Ẑcn₁,  Ẑcn₂]

              Ẑna = [Ẑn₁a, Ẑn₁b, Ẑn₁c]    Ẑnn = [Ẑn₁n₁, Ẑn₁n₂]
                    [Ẑn₂a, Ẑn₂b, Ẑn₂c]          [Ẑn₂n₁, Ẑn₂n₂]

        Definitions:
        Ẑ ----- "primative" impedance value, i.e. one that does not factor
                in the mutuals caused by neighboring neutral conductors.
        Z ----- a phase-phase impedance value that factors the mutual impedance
                of neighboring neutral conductors

        Returns:
        Z ----  a corrected impedance matrix in the form:

                     Zabc = [Zaa, Zab, Zac]
                            [Zba, Zbb, Zbc]
                            [Zca, Zcb, Zcc]
    """
    Ẑpp, Ẑpn = (z_primitive[0:dimension, 0:dimension],
                z_primitive[0:dimension, dimension:])
    Ẑnp, Ẑnn = (z_primitive[dimension:,  0:dimension],
                z_primitive[dimension:,  dimension:])
    Z_abc = Ẑpp - Ẑpn @ inv(Ẑnn) @ Ẑnp
    return Z_abc


def calculate_sequence_impedance_matrix(Z):
    return Ainv @ Z @ A


def calculate_sequence_impedances(Z):
    Z012 = calculate_sequence_impedance_matrix(Z)
    return Z012[1, 1], Z012[0, 0]


class CarsonsEquations():

    ρ = 100  # resistivity, ohms/meter^3
    μ = 4 * π * 1e-7  # permeability, Henry / meter

    def __init__(self, model):
        self.phases: Iterable[str] = model.phases
        self.phase_positions: Dict[str, Tuple[float, float]] = \
            model.wire_positions
        self.gmr: Dict[str, float] = model.geometric_mean_radius
        self.r: Dict[str, float] = model.resistance

        self.ƒ = getattr(model, 'frequency', 60)
        self.ω = 2.0 * π * self.ƒ  # angular frequency radians / second

    def build_z_primitive(self) -> ndarray:
        dimension = len(self.conductors)
        z_primitive = zeros(shape=(dimension, dimension), dtype=complex)

        for index_i, phase_i in enumerate(self.conductors):
            for index_j, phase_j in enumerate(self.conductors):
                if phase_i not in self.phases or phase_j not in self.phases:
                    continue
                R = self.compute_R(phase_i, phase_j)
                X = self.compute_X(phase_i, phase_j)
                z_primitive[index_i, index_j] = complex(R, X)

        return z_primitive

    def compute_R(self, i, j) -> float:
        rᵢ = self.r[i]
        ΔR = self.μ * self.ω / π * self.compute_P(i, j)

        if i == j:
            return rᵢ + ΔR
        else:
            return ΔR

    def compute_X(self, i, j) -> float:
        Qᵢⱼ = self.compute_Q(i, j)
        ΔX = self.μ * self.ω / π * Qᵢⱼ

        # calculate geometry ratio 𝛥G
        if i != j:
            Dᵢⱼ = self.compute_D(i, j)
            dᵢⱼ = self.compute_d(i, j)
            𝛥G = Dᵢⱼ / dᵢⱼ
        else:
            hᵢ = self.get_h(i)
            gmrⱼ = self.gmr[j]
            𝛥G = 2.0 * hᵢ / gmrⱼ

        X_o = self.ω * self.μ / (2 * π) * log(𝛥G)

        return X_o + ΔX

    def compute_P(self, i, j, number_of_terms=1) -> float:
        terms = islice(self.compute_P_terms(i, j), number_of_terms)
        return sum(terms)

    def compute_P_terms(self, i, j) -> Iterator[float]:
        yield π / 8.0

        kᵢⱼ = self.compute_k(i, j)
        θᵢⱼ = self.compute_θ(i, j)

        yield -kᵢⱼ / (3*sqrt(2)) * cos(θᵢⱼ)
        yield kᵢⱼ ** 2 / 16 * (0.6728 + log(2 / kᵢⱼ)) * cos(2 * θᵢⱼ)
        yield kᵢⱼ ** 2 / 16 * θᵢⱼ * sin(2 * θᵢⱼ)
        yield kᵢⱼ ** 3 / (45 * sqrt(2)) * cos(3 * θᵢⱼ)
        yield -π * kᵢⱼ ** 4 * cos(4 * θᵢⱼ) / 1536

    def compute_Q(self, i, j, number_of_terms=2) -> float:
        terms = islice(self.compute_Q_terms(i, j), number_of_terms)
        return sum(terms)

    def compute_Q_terms(self, i, j) -> Iterator[float]:
        yield -0.0386

        kᵢⱼ = self.compute_k(i, j)
        yield 0.5 * log(2 / kᵢⱼ)

        θᵢⱼ = self.compute_θ(i, j)
        yield kᵢⱼ / (3 * sqrt(2)) * cos(θᵢⱼ)
        yield -π * kᵢⱼ ** 2 / 64 * cos(2 * θᵢⱼ)
        yield kᵢⱼ ** 3 / (45 * sqrt(2)) * cos(3 * θᵢⱼ)
        yield -kᵢⱼ ** 4 / 384 * θᵢⱼ * sin(4 * θᵢⱼ)
        yield -kᵢⱼ ** 4 / 384 * cos(4 * θᵢⱼ) * (log(2 / kᵢⱼ) + 1.0895)

    def compute_k(self, i, j) -> float:
        Dᵢⱼ = self.compute_D(i, j)
        return Dᵢⱼ * sqrt(self.ω * self.μ / self.ρ)

    def compute_θ(self, i, j) -> float:
        xᵢ, _ = self.phase_positions[i]
        xⱼ, _ = self.phase_positions[j]
        xᵢⱼ = abs(xⱼ - xᵢ)
        hᵢ, hⱼ = self.get_h(i), self.get_h(j)

        return arctan(xᵢⱼ / (hᵢ + hⱼ))

    def compute_d(self, i, j) -> float:
        return self.calculate_distance(
            self.phase_positions[i],
            self.phase_positions[j],
        )

    def compute_D(self, i, j) -> float:
        xⱼ, yⱼ = self.phase_positions[j]

        return self.calculate_distance(self.phase_positions[i], (xⱼ, -yⱼ))

    @staticmethod
    def calculate_distance(positionᵢ, positionⱼ) -> float:
        xᵢ, yᵢ = positionᵢ
        xⱼ, yⱼ = positionⱼ
        return sqrt((xᵢ - xⱼ)**2 + (yᵢ - yⱼ)**2)

    def get_h(self, i):
        _, yᵢ = self.phase_positions[i]
        return yᵢ

    @property
    def dimension(self):
        return 2 if getattr(self, 'is_secondary', False) else 3

    @property
    def conductors(self):
        neutral_conductors = sorted([
            ph for ph in self.phases
            if ph.startswith("N")
        ])

        return ["A", "B", "C"] + neutral_conductors


class ModifiedCarsonsEquations(CarsonsEquations):
    """
    Modified Carson's Equation. Two approximations are made:
    only the first term of P and the first two terms of Q are considered.
    """
    number_of_P_terms = 1

    def compute_P(self, i, j, number_of_terms=1) -> float:
        return super().compute_P(i, j, self.number_of_P_terms)

    def compute_X(self, i, j) -> float:
        Q_first_term = super().compute_Q(i, j, 1)

        # Simplify equations and don't compute Dᵢⱼ explicitly
        kᵢⱼ_Dᵢⱼ_ratio = sqrt(self.ω * self.μ / self.ρ)
        ΔX = Q_first_term * 2 + log(2)

        if i == j:
            X_o = -log(self.gmr[i]) - log(kᵢⱼ_Dᵢⱼ_ratio)
        else:
            X_o = -log(self.compute_d(i, j)) - log(kᵢⱼ_Dᵢⱼ_ratio)

        return (X_o + ΔX) * self.ω * self.μ / (2 * π)


class ConcentricNeutralCarsonsEquations(ModifiedCarsonsEquations):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        self.neutral_strand_gmr: Dict[str, float] = model.neutral_strand_gmr
        self.neutral_strand_count: Dict[str, float] = defaultdict(
            lambda: None,
            model.neutral_strand_count
        )
        self.neutral_strand_resistance: Dict[str, float] = \
            model.neutral_strand_resistance
        self.radius: Dict[str, float] = defaultdict(
            lambda: None, {
                phase: (diameter_over_neutral -
                        model.neutral_strand_diameter[phase]) / 2
                for phase, diameter_over_neutral
                in model.diameter_over_neutral.items()
            }
        )
        self.phase_positions.update({
            f"N{phase}": self.phase_positions[phase]
            for phase in self.phase_positions.keys()
        })
        self.gmr.update({
            phase: self.GMR_cn(phase)
            for phase in model.diameter_over_neutral.keys()
        })
        self.r.update({
            phase: resistance / model.neutral_strand_count[phase]
            for phase, resistance in model.neutral_strand_resistance.items()
        })
        return

    def compute_d(self, i, j) -> float:
        I, J = set(i), set(j)
        r = self.radius[i] or self.radius[j]

        one_neutral_same_phase = I ^ J == set('N')
        different_phase = not I & J
        one_neutral = 'N' in I ^ J

        if one_neutral_same_phase:
            # Distance between a neutral/phase conductor of same phase
            return r

        distance_ij = self.calculate_distance(self.phase_positions[i],
                                              self.phase_positions[j])
        if different_phase and one_neutral:
            # Distance between a neutral/phase conductor of different phase
            # approximate by modelling the concentric neutral cables as one
            # equivalent conductor directly above the phase conductor
            return (distance_ij**2 + r**2) ** 0.5
        else:
            # Distance between two neutral/phase conductors
            return distance_ij

    def GMR_cn(self, phase) -> float:
        GMR_s = self.neutral_strand_gmr[phase]
        k = self.neutral_strand_count[phase]
        R = self.radius[phase]
        return (GMR_s * k * R**(k-1))**(1/k)


class TapeShieldedCableCarsonsEquations(ModifiedCarsonsEquations):

    ρ_tape_shield = 1.7721e-8  # copper resistivity at 20 degrees, ohm-meter

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        self.ds = model.tape_shield_outer_diameter
        self.thickness = model.tape_shield_thickness

        self.phases.extend(
            f'{ph}t' for ph in model.tape_shield_outer_diameter.keys()
        )

        self.r.update({
            f'{ph}t': self.compute_shield_r(self.ds[ph], self.thickness[ph])
            for ph in model.tape_shield_outer_diameter.keys()
        })

        self.gmr.update({
            f'{ph}t': self.compute_shield_gmr(self.ds[ph], self.thickness[ph])
            for ph in model.tape_shield_outer_diameter.keys()
        })

        # set shield position to be the same as its phase conductor position
        tape_shield_positions = {
            f'{ph}t': self.phase_positions[ph]
            for ph in model.tape_shield_outer_diameter.keys()
        }
        self.phase_positions.update(tape_shield_positions)

    @staticmethod
    def compute_shield_gmr(ds, thickness) -> float:
        return (ds - thickness) / 2

    @classmethod
    def compute_shield_r(cls, ds, thickness) -> float:
        area = (π*(ds/2)**2) - (π*(ds/2 - thickness)**2)
        return cls.ρ_tape_shield/area

    def compute_d(self, i, j) -> float:
        I, J = set(i), set(j)

        one_tape_shield_same_phase = I ^ J == set('t')

        if one_tape_shield_same_phase:
            return self.gmr[i] if 't' in i else self.gmr[j]
        else:
            return super().compute_d(i, j)

    @property
    def conductors(self):
        neutral_conductors = sorted([
            ph for ph in self.phases if ph.startswith("N")
        ])
        shield_conductors = sorted([
            ph for ph in self.phases if ph.endswith("t")
        ])

        return ['A', 'B', 'C'] + shield_conductors + neutral_conductors


class MultiConductorCarsonsEquations(ModifiedCarsonsEquations):
    def __init__(self, model):
        super().__init__(model)
        self.outside_radius: Dict[str, float] = model.outside_radius

    def compute_d(self, i, j) -> float:
        # Assumptions:
        # 1. All conductors in the cable are touching each other and
        #    therefore equidistant.
        # 2. In case of quadruplex cables, the space between conductors
        #    which are diagonally positioned is neglected.
        return self.outside_radius[i] + self.outside_radius[j]

    @property
    def conductors(self):
        neutral_conductors = sorted([
            ph for ph in self.phases if ph.startswith("N")
        ])
        if self.is_secondary:
            conductors = ["S1", "S2"] + neutral_conductors
        else:
            conductors = ["A", "B", "C"] + neutral_conductors

        return conductors

    @property
    def is_secondary(self):
        phase_conductors = [ph for ph in self.phases if not ph.startswith('N')]
        if phase_conductors == ["S1", "S2"]:
            return True
        else:
            return False
