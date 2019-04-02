from numpy import pi as π

from numpy import zeros
from numpy.linalg import inv
from numpy import sqrt
from numpy import log
from numpy import cos
from numpy import sin
from numpy import arctan
from itertools import islice


def convert_geometric_model(geometric_model):
    carsons_model = CarsonsEquations(geometric_model)

    z_primitive = carsons_model.build_z_primitive()
    z_abc = perform_kron_reduction(z_primitive)
    return z_abc


def perform_kron_reduction(z_primitive):
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
    Ẑpp, Ẑpn = z_primitive[0:3, 0:3], z_primitive[0:3, 3:]
    Ẑnp, Ẑnn = z_primitive[3:,  0:3], z_primitive[3:,  3:]
    Z_abc = Ẑpp - Ẑpn @ inv(Ẑnn) @ Ẑnp
    return Z_abc


class CarsonsEquations():

    ƒ = 60  # frequency, Hz
    ρ = 100  # resistivity, ohms/meter^3
    μ = 4 * π * 1e-7  # permeability, Henry / meter
    ω = 2.0 * π * ƒ  # angular frequency radians / second

    def __init__(self, model):
        self.phases = model.phases
        self.phase_positions = model.wire_positions
        self.gmr = model.geometric_mean_radius
        self.r = model.resistance

    def build_z_primitive(self):
        abc_conductors = [
            ph if ph in self.phases
            else None for ph in ("A", "B", "C")
        ]
        neutral_conductors = sorted([
            ph for ph in self.phases
            if ph.startswith("N")
        ])
        conductors = abc_conductors + neutral_conductors

        dimension = len(conductors)
        z_primitive = zeros(shape=(dimension, dimension), dtype=complex)

        for index_i, phase_i in enumerate(conductors):
            for index_j, phase_j in enumerate(conductors):
                if phase_i is not None and phase_j is not None:
                    R = self.compute_R(phase_i, phase_j)
                    X = self.compute_X(phase_i, phase_j)
                    z_primitive[index_i, index_j] = complex(R, X)

        return z_primitive

    def compute_R(self, i, j):
        rᵢ = self.r[i]
        ΔR = self.μ * self.ω / π * self.compute_P(i, j)

        if i == j:
            return rᵢ + ΔR
        else:
            return ΔR

    def compute_X(self, i, j):
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

    def compute_P(self, i, j, number_of_terms=1):
        terms = islice(self.compute_P_terms(i, j), number_of_terms)
        return sum(terms)

    def compute_P_terms(self, i, j):
        yield π / 8.0

        kᵢⱼ = self.compute_k(i, j)
        θᵢⱼ = self.compute_θ(i, j)

        yield -kᵢⱼ / (3*sqrt(2)) * cos(θᵢⱼ)
        yield kᵢⱼ ** 2 / 16 * (0.6728 + log(2 / kᵢⱼ)) * cos(2 * θᵢⱼ)
        yield kᵢⱼ ** 2 / 16 * θᵢⱼ * sin(2 * θᵢⱼ)
        yield kᵢⱼ ** 3 / (45 * sqrt(2)) * cos(3 * θᵢⱼ)
        yield -π * kᵢⱼ ** 4 / 64 * cos(2 * θᵢⱼ)

    def compute_Q(self, i, j, number_of_terms=2):
        terms = islice(self.compute_Q_terms(i, j), number_of_terms)
        return sum(terms)

    def compute_Q_terms(self, i, j):
        yield -0.0386

        kᵢⱼ = self.compute_k(i, j)
        yield 0.5 * log(2 / kᵢⱼ)

        θᵢⱼ = self.compute_θ(i, j)
        yield kᵢⱼ / (3 * sqrt(2)) * cos(θᵢⱼ)
        yield -π * kᵢⱼ ** 2 / 64 * cos(2 * θᵢⱼ)
        yield kᵢⱼ ** 3 / (45 * sqrt(2)) * cos(3 * θᵢⱼ)
        yield -kᵢⱼ ** 4 / 384 * θᵢⱼ * sin(4 * θᵢⱼ)
        yield -kᵢⱼ ** 4 / 384 * cos(4 * θᵢⱼ) * (log(2 / kᵢⱼ) + 1.0895)

    def compute_k(self, i, j):
        Dᵢⱼ = self.compute_D(i, j)
        return Dᵢⱼ * sqrt(self.ω * self.μ / self.ρ)

    def compute_θ(self, i, j):
        xᵢ, _ = self.phase_positions[i]
        xⱼ, _ = self.phase_positions[j]
        xᵢⱼ = abs(xⱼ - xᵢ)
        hᵢ, hⱼ = self.get_h(i), self.get_h(j)

        return arctan(xᵢⱼ / (hᵢ + hⱼ))

    def compute_d(self, i, j):
        return self.calculate_distance(
            self.phase_positions[i],
            self.phase_positions[j])

    def compute_D(self, i, j):
        xⱼ, yⱼ = self.phase_positions[j]
        return self.calculate_distance(self.phase_positions[i], (xⱼ, -yⱼ))

    @staticmethod
    def calculate_distance(positionᵢ, positionⱼ):
        xᵢ, yᵢ = positionᵢ
        xⱼ, yⱼ = positionⱼ
        return sqrt((xᵢ - xⱼ)**2 + (yᵢ - yⱼ)**2)

    def get_h(self, i):
        _, yᵢ = self.phase_positions[i]
        return yᵢ


class ConcentricNeutralCarsonsEquations(CarsonsEquations):
    def __init__(self, *args, **kwargs):
        return

    @property
    def impedance(self):
        return zeros(shape=(3, 3), dtype=complex)
