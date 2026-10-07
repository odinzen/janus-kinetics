# Theory

The equations calphad-kinetics implements, with the paper and equation number
each one comes from. Full references are in [references.md](references.md).
Compositions are mole fractions x_k of n elements with the dependent element
last; all phases are substitutional with one molar volume V_m unless stated.

## Atomic mobility

For each diffusing element B the database holds MF = RT ln M0_B and MQ = -Q_B,
each expanded over composition with end-member values, Redlich-Kister binary
terms and a ternary term (Andersson and Agren 1992, eq. 67; Jonsson 1995,
eqs. 5-6):

    Phi_B = sum_i x_i Phi_B^i + sum_{i<j} x_i x_j sum_r Phi_B^{ij,r} (x_i - x_j)^r
            + sum_{i<j<k} x_i x_j x_k sum_s v_ijk^s Phi_B^{ijk,s},
    v_ijk^s = x_s + (1 - x_i - x_j - x_k)/3

    M_B = exp((MF_B + MQ_B)/RT) / RT          (Jonsson 1995, eqs. 3-4)
    D*_B = RT M_B                             (Andersson and Agren 1992, eq. 61)

The ferromagnetic factor of Jonsson's eq. 3 is a caller-supplied multiplier
(default 1) until its defining paper (Jonsson 1992) is checked. The evaluation
uses pycalphad's own Redlich-Kister machinery, so interaction parameters follow
pycalphad's ordering. Constituents are sorted alphabetically, and an odd-order
term written as `B,A;1` multiplies (x_A - x_B).

## Thermodynamic factor

The molar Gibbs energy G comes symbolically from `pycalphad.Model.GM` and is
differentiated exactly. With x_n eliminated, the curvature is
G''_kj = H_kj - H_kn - H_nj + H_nn, and the chemical potentials and their
derivatives are

    mu_i = G + dG/dx_i - sum_k x_k dG/dx_k
    dmu_i/dx_j = sum_k (delta_ik - x_k) G''_kj      (j < n)

If G'' is not positive definite the composition lies inside a spinodal and the
single-phase models stop with `UnstableCompositionError`.

## Interdiffusion

Lattice-fixed fluxes of the vacancy mechanism are J'_k = -c_k M_k grad mu_k
(Andersson and Agren 1992, eqs. 58 and 62). At constant V_m they define the
intrinsic diffusivities

    D^I_kj = x_k M_k dmu_k/dx_j

and the number-fixed interdiffusion matrix (eqs. 59-60; quoted as eq. 2 of
Jonsson 1995)

    D_kj = sum_i (delta_ik - x_k) x_i M_i dmu_i/dx_j

The volume-fixed matrix with constant partial molar volumes V_i follows from
eqs. 28-29, 40, 50, 54 and 58 of Andersson and Agren:

    D^V_kj = sum_i (delta_ik - c_k V_i) c_i M_i dmu_i/dc_j,
    dx_k/dc_j = V_m [delta_kj - x_k (1 - V_j/V_n)]

## Planar solver

dx_k/dt = d/dz (sum_j D_kj dx_j/dz) is solved by backward Euler on a finite-volume
grid, with D at each face from the mean of the neighbouring cells and Picard
iteration for its composition dependence. The face-flux form conserves the
inventory of every element under no-flux boundaries, to floating-point round-off.

## Kirkendall markers

The number-fixed and lattice frames are related by J_k = J'_k + c_k v (van Loo
1990, eq. 4). With sum_k J_k = 0 the lattice velocity is

    v = sum_k sum_j D^I_kj dx_j/dz

which for a binary is v = (D_A - D_B) dx_A/dz (eq. 9). Markers are moved by
dz/dt = v, integrated in sqrt(t), where the parabolic motion of a couple
(eq. 5) is constant.

## Short-circuit diffusion

An atom spending a fraction f of its time in fast paths sees
D_eff = f D_path + (1 - f) D_lattice (Hart 1957, eq. 5, derived for
self-diffusion along dislocations). Applying the same average to grain
boundaries, and through D* = RT M to every element's mobility, is this
package's extension of Hart's argument.

## Isolated grain boundaries

Fisher (1951): a boundary slab of thickness delta and diffusivity D' normal to a
surface held at C = 1, in grains of diffusivity D. In the scaled variables of his
eq. 5 (t1 = D t / delta^2, x1 = x / delta, y1 = y / (delta (D'/D)^1/2)):

    dC/dt1 = d2C/dy1^2 + 2 (dC/dx1)_0     (slab, eq. 3a)
    dC/dt1 = d2C/dx1^2                    (grains, eq. 4a)

His approximate solution is C = exp(-2^1/2 y1 / (pi^1/4 t1^1/4)) erfc(x1 / 2 t1^1/2)
(eq. 8), whose layer average gives the penetration slope of eq. 14. The package
also solves eqs. 3a-4a numerically; that solution agrees with the exact Laplace
transform solution, C~ = exp(-y1 (s + 2 s^1/2)^1/2 - x1 s^1/2) / s, and shows eq. 8
overestimating penetration at depth.

## Interstitial elements

Andersson and Agren 1992, sections II.B and VI.A. With u_k = x_k / sum_{j in S} x_j
(eq. 9), equal to y on the substitutional sublattice and (a^I / a^S) y^I for an
interstitial (eq. 10), the lattice fluxes are J'_k = -c_k Omega_k grad mu_k for
substitutional elements and -c_k y_Va Omega_k grad mu_k for interstitials
(eqs. 62-63), and the diffusivity matrix in the frame fixed to the substitutional
elements is

    D_kj = sum_{i in S} (delta_ik - u_k) u_i Omega_i dmu_i/du_j + [k interstitial] u_k y_Va Omega_k dmu_k/du_j

reduced to the independent u-fractions by eq. 65. Chemical potentials come from
pycalphad's Gibbs energy per mole of substitutional atoms G_u, as mu_k = dG_u/du_k
for an interstitial and G_u + dG_u/du_k - sum_j u_j dG_u/du_j for a substitutional
element.

## Moving interface

Two phases share a planar interface at S(t). Each is solved on its own grid,
stretched with S, where the face flux is F = J - w x_up for a face moving at w.
Every independent element satisfies the interface flux balance, Zener's eq. 8
(Zener 1949) generalized to diffusion on both sides:

    dS/dt (x_R^I - x_L^I) = J_R^I - J_L^I

With local equilibrium the interface compositions also make every chemical
potential equal across the interface. That is n equations, so in a binary the
tie-line is fixed and only dS/dt is solved for; with more elements the operating
tie-line and dS/dt are solved together each step.

## Stoichiometric reaction layers

A line compound has no measurable gradient, so its flux is described by Wagner's
integrated diffusion coefficient (van Loo 1990, eq. 17), D_int = integral of D~ dN
across the layer. With D~ = x_A D_B + x_B D_A (eq. 10 at equal partial volumes)
and D_i = M_i x_i dmu_i/dx_i, a layer of thickness d and molar volume V carries the
intrinsic fluxes

    J_A = x_A x_B M_A dS / (V d),    J_B = -x_A x_B M_B dS / (V d)
    D_int = x_A x_B (x_A M_B + x_B M_A) dS

where dS is the jump of mu_B - mu_A across it. Local equilibrium puts mu_B - mu_A
at each interface on the chord between the two neighbouring compounds, so dS is
the difference of the chord slopes at the layer, positive when the compound is
below the line joining its neighbours. Each layer's lattice moves rigidly, and
conservation of each element across each moving interface,

    c_R (v - w_R) - c_L (v - w_L) = J_R - J_L,

gives every interface velocity v and lattice velocity w, solved outward from one
end member. Markers move with the lattice they sit in. In a fresh couple every
length grows as sqrt(t) (van Loo p. 61); the growth constants solve
delta_k = 2 (v_k - v_(k-1)) with v computed at thicknesses delta, reached by
relaxing in ln t and polishing with Newton. A finite end member that is used up
hands its role to the next phase. The model reproduces van Loo's Ni3Sn2 analysis
(pp. 64-65: consumption ratio 125/103, marker split 0.5 or 2 when one element
moves, and the 3.4 um marker shift between D_Ni/D_Sn = 5 and infinity) and the
wustite couples of p. 78 (d1 = 2 d2). Interface-reaction control and nucleation
barriers are not included.

## Homogenization

Each cell is in local equilibrium at its overall composition, so it has one set
of chemical potentials whatever phases it holds (Larsson and Hoglund 2009):

    J_k = -(1/V_S) Gamma*_k dmu_k/dz,       Gamma^phi_k = M^phi_k x^phi_k     (eqs. 2-5)

Gamma* is a Wiener bound (eqs. 6-7) or a Hashin-Shtrikman bound (eqs. 8-11) of
the phases' Gamma^phi weighted by their volume fractions. Local equilibrium is
tabulated once per temperature with `pycalphad.equilibrium` and interpolated
linearly. The paper (section 2.5) fills such a table on demand instead, because a
complete one is impractical beyond three elements; its ideal flux term against
oscillations (section 2.4, eq. 13) is not included. For anneals in binaries a
table is built at each of several temperatures; between them chemical potentials
are interpolated linearly in T and ln Gamma* linearly in 1/T, which is exact for
Arrhenius mobilities. Face fluxes use the difference of chemical potentials
between cells, so transport across a phase boundary is driven by the actual
potential jump.

## Precipitation

Perez, Dumont and Acevedo-Reyes 2008, with eqs. 4, 7 and note 3 as corrected in
the 2009 corrigendum. Spherical precipitates of solute fraction X^P grow in a
matrix of solute fraction X.

    X^i(R): (1 - X^P) ln X_A^i(R) + X^P ln X_B^i(R)
            = (1 - X^P) ln X_A^i + X^P ln X_B^i + R0/R,   R0 = 2 gamma v_at^P / kT   (eq. 9)
    dg = -(kT/v_at^P) [(1 - X^P) ln((1 - X)/(1 - X^i)) + X^P ln(X/X^i)]
    R* = -2 gamma / dg,    dG* = 16 pi gamma^3 / (3 dg^2)                        (eq. 1)
    Z = v_at^P / (2 pi R*^2) sqrt(gamma / kT)                                    (eq. 4, corrected)
    beta* = 4 pi R*^2 D X / a^4                                                  (eq. 5)
    dN/dt = N0 Z beta* exp(-dG*/kT) exp(-tau/t),  tau = 4 / (2 pi beta* Z^2)     (eq. 6)
    R_nuc = R* + (1/2) sqrt(kT / (pi gamma))                                     (eq. 7, corrected)
    dR/dt = (D/R) (X - X^i(R)) / (alpha X^P - X^i(R)),  alpha = v_at^M / v_at^P  (eq. 10)
    X0 = alpha f X^P + X (1 - alpha f),  f = (4/3) pi sum N_i R_i^3              (eqs. 21-22)

The driving force is not written out in the paper; the form above is the one
consistent with eq. 9, since a nucleus of radius R* then has exactly the matrix
composition at its interface. The size distribution lives in fixed classes that
exchange particles through upwind fluxes at their edges (eq. 24), solved
implicitly. Steps grow by 10% and are halved when R* changes by more than 1%,
the rule of section 3.4.1 (given there for the Lagrange-like approach); the same
limit on X and on the mean radius is this package's addition.
