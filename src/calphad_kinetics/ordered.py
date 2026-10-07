"""Diffusion in B2-ordered bcc, after Helander and Agren, Acta Mater. 47 (1999) 1141.

Chemical order raises the activation energy for diffusion. Their eqs. 11-12 add to
the activation energy of element k in the disordered state

    dQ_k^ord = sum_i sum_(j != i) dQ^order_kij [ y_i^a y_j^b - x_i x_j ]

where y^a and y^b are the site fractions on the two B2 sublattices and x_i their
mean, vacancies included as a constituent. In a binary without vacancies this is
Girifalco's dQ = S^2 dQ^order / 2 with S = y_A^a - y_A^b (their Appendix B). The
mobility is then M_k = exp((Phi_k^dis - dQ_k^ord) / RT) / RT with Phi^dis the
usual disordered Andersson-Agren expansion at the mole fractions.

Site fractions and chemical potentials come from ``pycalphad.equilibrium`` of the
ordered phase alone at each composition, so order relaxes as the composition
changes; the thermodynamic factor is taken by central differences of the
equilibrium chemical potentials. Each evaluation is an equilibrium calculation,
so tabulate before using these in a solver loop.

Validated against the paper's Al-Ni Figs. 2 and 7 for x_Al <= 0.5 to within 0.04
decades. On the Al-rich side, where structural vacancies form, the chemical
diffusivity comes out about 0.3 decades above their Fig. 2 at x_Al = 0.53-0.54.
The thermodynamics is not the cause: the Al-Ni B2 parameters used in the tests
are those of Ansara et al. (1997), which the paper used, as printed by Dupin et
al. (2001). Taking x_i in eq. 12 as atom fractions instead of site averages makes
it three decades worse, so how the paper treats vacancies remains open, and a
vacancy site fraction above 1 % raises a warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from pycalphad import Database, Model, equilibrium, variables as v

from ._lattice import SubstitutionalLattice
from .diffusivity import number_fixed_matrix
from .errors import UnsupportedPhaseError
from .mobility import MobilityModel
from .thermo import R


def ordering_energy(terms: Mapping[tuple[str, str, str], float], element: str, constituents, y_a, y_b) -> np.ndarray:
    """Eq. 12: dQ^ord (J/mol) of ``element`` for site fractions y_a, y_b over ``constituents``.

    ``terms`` maps (k, i, j) to dQ^order_kij, with i and j in either order. The two
    sublattices must have equal site numbers, as in B2, so x_i = (y_i^a + y_i^b) / 2.
    """
    y_a, y_b = np.asarray(y_a, dtype=float), np.asarray(y_b, dtype=float)
    x = 0.5 * (y_a + y_b)
    total = np.zeros(y_a.shape[:-1])
    for p, i in enumerate(constituents):
        for q, j in enumerate(constituents):
            if p == q:
                continue
            dq = terms.get((element, i, j), terms.get((element, j, i), 0.0))
            if dq:
                total = total + dq * (y_a[..., p] * y_b[..., q] - x[..., p] * x[..., q])
    return total


class B2Phase:
    """Mobilities and diffusivities of a B2-ordered phase.

    ``db`` holds the Gibbs energy of ``phase_name``, two sublattices of equal site
    number holding the elements (and possibly vacancies) plus any vacancy-only
    sublattices. ``mobility_db`` holds the disordered mobilities of
    ``disordered_phase``; ``ordering`` the dQ^order_kij terms. Composition
    arguments follow the package convention: mole fractions of ``elements`` with
    the dependent element last, vacancies excluded.
    """

    def __init__(self, db: Database, phase_name: str, elements, mobility_db: Database,
                 disordered_phase: str, ordering: Mapping[tuple[str, str, str], float], dependent: str | None = None):
        self.db = db
        self.phase_name = phase_name.upper()
        self.disordered = MobilityModel(
            mobility_db, SubstitutionalLattice.from_database(mobility_db, disordered_phase, elements, dependent))
        self.elements = self.disordered.lattice.elements
        self.ordering = {tuple(str(s).upper() for s in key): float(value) for key, value in ordering.items()}
        unknown = {s for key in self.ordering for s in key} - set(self.elements) - {"VA"}
        if unknown:
            raise ValueError(f"ordering terms name {sorted(unknown)}, which are not among {self.elements} or VA")

        phase = db.phases.get(self.phase_name)
        if phase is None:
            raise UnsupportedPhaseError(f"phase {self.phase_name} is not in the database")
        wanted = set(self.elements)
        mixing = [k for k, subl in enumerate(phase.constituents) if wanted <= {str(s.name) for s in subl}]
        if len(mixing) != 2 or phase.sublattices[mixing[0]] != phase.sublattices[mixing[1]]:
            raise UnsupportedPhaseError(
                f"{self.phase_name} needs exactly two sublattices of equal site number holding {sorted(wanted)}")
        self.components = list(self.elements) + ["VA"]
        model = Model(db, self.components, self.phase_name)
        self._sites = [[str(y.species.name) for y in model.site_fractions if y.sublattice_index == k] for k in mixing]
        self._offsets = []
        start = 0
        for k, subl in enumerate(model.constituents):
            if k in mixing:
                self._offsets.append(start)
            start += len(subl)
        self.constituents = tuple(dict.fromkeys(self._sites[0] + self._sites[1]))

    def _x(self, x) -> np.ndarray:
        return self.disordered.lattice.full_composition(x)

    def _equilibrium(self, x_full: np.ndarray, T: float):
        conds = {v.T: T, v.P: 101325, v.N: 1}
        conds.update({v.X(el): float(xi) for el, xi in zip(self.elements[:-1], x_full[:-1])})
        eq = equilibrium(self.db, self.components, [self.phase_name], conds)
        fractions = np.nan_to_num(eq.NP.values.squeeze())
        if np.count_nonzero(fractions > 1e-10) > 1:
            raise UnsupportedPhaseError(f"{self.phase_name} separates into two compositions at x = {x_full} and {T:g} K")
        return eq

    def site_fractions(self, x, T: float) -> tuple[np.ndarray, np.ndarray]:
        """Equilibrium site fractions (y_a, y_b) over :attr:`constituents` at one composition."""
        eq = self._equilibrium(self._x(x), T)
        y = eq.Y.values.squeeze()[0]
        out = []
        for names, offset in zip(self._sites, self._offsets):
            row = np.zeros(len(self.constituents))
            for n, name in enumerate(names):
                row[self.constituents.index(name)] = y[offset + n]
            out.append(row)
        return out[0], out[1]

    def chemical_potentials(self, x, T: float) -> np.ndarray:
        """Equilibrium chemical potentials of :attr:`elements` (J/mol) with order relaxed."""
        eq = self._equilibrium(self._x(x), T)
        mu = dict(zip(eq.component.values, eq.MU.values.squeeze()))
        return np.array([mu[el] for el in self.elements])

    def mobility(self, x, T: float) -> np.ndarray:
        """Atomic mobilities M_k in m^2 mol / (J s), in the order of :attr:`elements`."""
        x_full = self._x(x)
        y_a, y_b = self.site_fractions(x_full, T)
        if "VA" in self.constituents:
            vacancies = 0.5 * (y_a + y_b)[self.constituents.index("VA")]
            if vacancies > 0.01:
                warnings.warn(f"structural vacancy fraction {vacancies:.3f} at x = {x_full}: the vacancy ordering "
                              "terms are not validated against Helander and Agren's figures", stacklevel=2)
        d_q = np.array([ordering_energy(self.ordering, el, self.constituents, y_a, y_b) for el in self.elements])
        rt = R * T
        return np.exp((self.disordered.phi(x_full, T) - d_q) / rt) / rt

    def tracer_diffusivity(self, x, T: float) -> np.ndarray:
        """D*_k = RT M_k in m^2/s, without the correlation factor."""
        return R * T * self.mobility(x, T)

    def interdiffusion_matrix(self, x, T: float, step: float = 1e-3) -> np.ndarray:
        """Number-fixed interdiffusion matrix (m^2/s), dependent element last.

        dmu_i/dx_j by central differences of :meth:`chemical_potentials` with
        ``step`` in each independent mole fraction, the dependent one taking up the
        change.
        """
        x_full = self._x(x)
        n = len(self.elements)
        dmu = np.empty((n, n - 1))
        for j in range(n - 1):
            dx = np.zeros(n)
            dx[j], dx[-1] = step, -step
            dmu[:, j] = (self.chemical_potentials(x_full + dx, T) - self.chemical_potentials(x_full - dx, T)) / (2 * step)
        return number_fixed_matrix(x_full, self.mobility(x_full, T), dmu)
