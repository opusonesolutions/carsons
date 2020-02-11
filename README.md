carsons
=======

[![latest release on pypi](https://badge.fury.io/py/carsons.svg)](https://badge.fury.io/py/carsons)
[![versons of python supported by carsons](https://img.shields.io/pypi/pyversions/carsons.svg)](https://pypi.python.org/pypi/carsons)
[![GitHub license](https://img.shields.io/github/license/opusonesolutions/carsons.svg)](https://github.com/opusonesolutions/carsons/blob/master/LICENSE.txt)
[![build passing or failing](https://travis-ci.org/opusonesolutions/carsons.svg?branch=master)](https://travis-ci.org/opusonesolutions/carsons)
[![test coverage](https://coveralls.io/repos/github/opusonesolutions/carsons/badge.svg?branch=master)](https://coveralls.io/github/opusonesolutions/carsons?branch=master)
[![Maintainability](https://api.codeclimate.com/v1/badges/22cfed180fd6032fe29b/maintainability)](https://codeclimate.com/github/opusonesolutions/carsons/maintainability)

This is an implementation of Carson's Equations, a mathematical model
for deriving the equivalent impedance of an AC transmission or
distribution line.

Implementation
--------------

`carsons` is developed using python 3.6 support for unicode characters
like π, ƒ, ρ, μ, ω etc. This feature allows us to avoid translating the
problem into a more typical programming syntax, so the code is dense and
can easily be compared to published formulations of the problem.

For example, we implement the kron reduction, a matrix decomposition
step, using unicode notation to indicate the slightly different meaning
of impedance values before and after a kron reduction:

```python
def perform_kron_reduction(z_primitive):
     Ẑpp, Ẑpn = z_primitive[0:3, 0:3], z_primitive[0:3, 3:]
     Ẑnp, Ẑnn = z_primitive[3:,  0:3], z_primitive[3:,  3:]
     Z_abc = Ẑpp - Ẑpn @ inv(Ẑnn) @ Ẑnp
     return Z_abc
```

Take a look at the [source
code](https://github.com/opusonesolutions/carsons/blob/add-documentation/carsons/carsons.py)
to see more cool unicode tricks!

Installation
------------

```bash
~/$ pip install carsons
```

Usage
-----

Carsons model requires a line model object that maps each phase to
properties of the conductor for that phase.

```python
from carsons import CarsonsEquations, calculate_impedance

class Line:
   geometric_mean_radius: {
       'A': geometric_mean_radius_A in meters
       ...
   }
   resistance: {
        'A': per-length resistance of conductor A in ohms/meters
        ...
   }
   wire_positions: {
        'A': (x, y) cross-sectional position of the conductor in meters
        ...
   }
   phases: {'A', ... }
     # map of phases 'A', 'B', 'C' and 'N<>' which are described in the
     # gmr, r and phase_positions attributes

line_impedance = calculate_impedance(CarsonsEquations(Line()))
```

The model supports any combination of ABC phasings (for example BC, BCN
etc...) including systems with multiple neutral cables; any phases that
are not present in the model will have zeros in the columns and rows
corresponding to that phase.

Multiple neutrals are supported, as long as they have unique labels
starting with `N` (e.g. `Neutral1`, `Neutral2`).

Intermediate results such as primitive impedance matrix are also
available.

```python
z_primitive = CarsonsEquations(Line()).build_z_primitive()
```

For examples of how to use the model, see the [overhead wire
tests](https://github.com/opusonesolutions/carsons/blob/master/tests/test_overhead_line.py).

`carsons` is tested against several cable configurations from the [IEEE
test feeders](http://sites.ieee.org/pes-testfeeders/resources/), as well as
examples from  EPRI's [OpenDSS documentation](http://svn.code.sf.net/p/electricdss/code/trunk/Distrib/Doc/TechNote%20CableModelling.pdf).

### Concentric Neutral Cable

`carsons` also supports modelling of concentric neutral cables of any
phasings. Its usage is very similar to the example above, only requiring
a few more parameters about the neutral conductors in the line model
object.

```python
from carsons import (ConcentricNeutralCarsonsEquations,
                     calculate_impedance)

class Cable:
   resistance: {
       'A': per-length resistance of conductor A in ohm/meters
       ...
   }
   geometric_mean_radius: {
       'A': geometric mean radius of conductor A in meters
       ...
   }
   wire_positions: {
        'A': (x, y) cross-sectional position of conductor A in meters
        ...
   }
   phases: {'A', 'NA', ... }
   neutral_strand_gmr: {
       'NA': neutral strand gmr of phase A in meters
       ...
   }
   neutral_strand_resistance: {
       'NA': neutral strand resistance of phase A in ohm/meters
       ...
   }
   neutral_strand_diameter: {
       'NA': neutral strand diameter of phase A in meters
       ...
   }
   diameter_over_neutral: {
       'NA': diameter over neutral of phase A in meters
       ...
   }
   neutral_strand_count: {
       'NA': neutral strand count of phase A
       ...
   }

cable_impedance = calculate_impedance(ConcentricNeutralCarsonsEquations(Cable()))
```

For examples of how to use the model, see the [concentric cable
tests](https://github.com/opusonesolutions/carsons/blob/master/tests/test_concentric_neutral_cable.py).

### Multi-Conductor Cable

`carsons` also supports modelling of phased duplex, triplex, quadruplex cables and triplex secondary.
It only requires a few more parameters to describe cable's geometry.

```python
from carsons import (MultiConductorCarsonsEquations,
                     calculate_impedance)

class Cable:
    resistance: {
        'A': per-length resistance of conductor A in ohm/meters
        ...
    }
    geometric_mean_radius: {
        'A': geometric mean radius of conductor A in meters
        ...
    }
    wire_positions: {
        'A': (x, y) cross-sectional position of conductor A in meters
        ...
    }
    radius: {
        'A': radius of conductor A
        ...
    }
    insulation_thickness: {
        'A': insulation thickness of conductor A
        ...
    }
    phases: {'A', ... }

cable_impedance = calculate_impedance(MultiConductorCarsonsEquations(Cable()))
```

To model a triplex secondary cable, the inputs should be keyed on secondary conductors `S1` and `S2`. The impedance result
is a 2 x 2 matrix.

```python
class Cable:
    resistance: {
        'S1': per-length resistance of conductor S1 in ohm/meters
        ...
    }
    geometric_mean_radius: {
        'S1': geometric mean radius of conductor S1 in meters
        ...
    }
    wire_positions: {
        'S1': (x, y) cross-sectional position of conductor S1 in meters
        ...
    }
    radius: {
        'S1': radius of conductor S1
        ...
    }
    insulation_thickness: {
        'S1': insulation thickness of conductor S1
        ...
    }
    phases: {'S1', ... }
```

For examples of how to use the model, see the [multi-conductor cable
tests](https://github.com/opusonesolutions/carsons/blob/master/tests/test_multi_conductor.py).

Problem Description
-------------------

Carsons equations model an AC transmission or distribution line into an
equivalent set of phase-phase impedances, which can be used to model the
line in a power flow analysis.

For example, say we have a 4-wire system on a utility pole, with `A`,
`B`, `C` phase conductors as well as a neutral cable N. We know that
when conductors carry electrical current, they exhibit a magnetic field
--- so its pretty easy to imagine that, e.g., the magnetic field
produced by `A` would interact with the `B`, `C`, and `N` conductors.

                            B
                              O
                              |
                              |
                  A        N  |       C
                    O        O|         O
                    ----------|-----------
                              |
                              |
                              |
                              |
                              |
                              |
                              |
                              |
                              |
                              |
                              |
                              |
                              |
        ==============[Ground]============================
        /     /     /     /     /     /     /     /     /
             /     /     /     /     /     /     /
                  /     /     /     /     /
     
     
     
     
     
     
     
     
     
     
                     A*       N*          C*
                       0        0           0
     
                               B*
                                 0

    Figure: Cross-section of a 4-wire distribution line, with
            ground return.

However, each conductor also has a ground return path (or 'image') ---
shown as `A*`, `B*`, `C*`, and `N*` in the figure above --- which is a
magnetically induced current path in the ground. When A produces a
magnetic field, that field *also* interacts with `B*`, `C*`, `N*`, *and*
`A*`. Carsons equations model all these interactions and reduce them to
an equivalent impedance matrix that makes it much easier to model this
system.

In addition `carsons` implements the kron reduction, a conversion that
approximates the impedances caused by neutral cables by incorporating
them into the impedances for phase `A`, `B`, and `C`. Since most AC and
DC powerflow formulations don't model the neutral cable, this is a
valuable simplification.

References
----------

The following works were used to produce this formulation:

-   [Leonard L. Grigsby -- Electrical Power Generation, Transmission and
    Distribution](https://books.google.ca/books?id=XMl8OU4wIEQC&lpg=SA21-PA4&dq=kron%20reduction%20carson%27s%20equation&pg=SA21-PA4#v=onepage&q=kron%20reduction%20carson's%20equation&f=true)
-   [William H. Kersting -- Distribution System Modelling and Analysis
    2e](https://books.google.ca/books?id=1R2OsUGSw_8C&lpg=PA84&dq=carson%27s%20equations&pg=PA85#v=onepage&q=carson's%20equations&f=false)
-   [William H. Kersting, Distribution System Analysis Subcommittee --
    Radial Distribution Test
    Feeders](http://sites.ieee.org/pes-testfeeders/files/2017/08/testfeeders.pdf)
-   [Timothy Vismore -- The Vismor
    Milieu](https://vismor.com/documents/power_systems/transmission_lines/S2.SS1.php)
-   [Daniel Van Dommelen, Albert Van Ranst, Robert Poncelet -- GIC
    Influence on Power Systems calculated by Carson's
    method](https://core.ac.uk/download/pdf/34634673.pdf)
-   [Andrea Ballanti, Roger Dugan -- Cable Modelling in OpenDSS](http://svn.code.sf.net/p/electricdss/code/trunk/Distrib/Doc/TechNote%20CableModelling.pdf)
