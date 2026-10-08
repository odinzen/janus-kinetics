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

Since sum_i V_i c_i = 1 everywhere, the volume-fixed frame is fixed to the ends
of a closed sample, and the solver can integrate the concentrations c_k on a
fixed grid with D^V, conserving moles while V_m varies with composition. The
lattice moves relative to that frame at v = -sum_k V_k J'_k (van Loo 1990,
eqs. 4 and 6), which in a binary is eq. 9, v = V_A (D_A - D_B) dc_A/dz, with
D~ = c_A V_A D_B + c_B V_B D_A (eq. 10).

## Reading a measured couple

The inverse problem: from a binary penetration curve after time t, the
interdiffusion coefficient follows without a Matano plane (van Loo eq. 3),

    D~(Y*) = V_m/(2t) (dz/dY)* [ (1 - Y*) int_{-inf}^{z*} Y/V_m dz + Y* int_{z*}^{inf} (1 - Y)/V_m dz ],

with Y = (N - N^-)/(N^+ - N^-), and the intrinsic diffusivities at the marker
plane follow from eq. 12,

    D_i(z_K) = 1/(2t) (dz/dC_i)_K [ N_i^+ int_{-inf}^{z_K} Y/V_m dz - N_i^- int_{z_K}^{inf} (1 - Y)/V_m dz ].

Both reduce to the constant-volume eqs. 2 and 11. Applied to a simulated couple
with V_B/V_A = 1.8 they recover the model's D~ to 0.5 % and both intrinsic
diffusivities to 1 %; ignoring the volume change misses D_A by over 10 %.

## Planar solver

dx_k/dt = d/dz (sum_j D_kj dx_j/dz) is solved by backward Euler on a finite-volume
grid, with D at each face from the mean of the neighbouring cells and Picard
iteration for its composition dependence. The face-flux form conserves the
inventory of every element under no-flux boundaries, to floating-point round-off.

## Cylinders and spheres

The planar solver and the layered moving-interface solver also run in a long
cylinder or a sphere, the one-variable geometries of DICTRA (Borgenstam et al.
2000): dx/dt = r^-p d/dr (r^p D dx/dr) with p = 1 or 2. Cells hold
(r_out^(p+1) - r_in^(p+1)) / (p+1) per unit angle and faces carry r^p; the axis or
centre takes no flux. A face that moves sweeps the volume between its old and new
radius, and its advective term uses that swept volume over the distance moved as its
area, which keeps a uniform composition uniform on a stretching grid (the core of a
growing particle stays exactly at its composition). End fluxes are returned per unit
of that area, so the interface balance w (x_R - x_L) = J_R - J_L keeps its planar
form and every element is conserved to round-off. Checks: series solutions for a
sphere and a cylinder with the surface held (profiles within 0.03 % of the change,
first order in time), and Zener's eq. 12 for cylinder (lambda = 2) and sphere
(lambda = 3) growth, with his integrals in closed form; the particle radius at 100
times the start time is within 0.24 % and 0.35 %, falling with the grid.

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
wustite couples of p. 78 (d1 = 2 d2). For several layers grown from pure end
members, Wagner's eq. 20 (Acta Metall. 17 (1969) 99) gives each layer's D_int from
the widths of all of them, weighted by the ratios of molar volumes; the simulated
widths satisfy it to round-off. Interface-reaction control and nucleation
barriers are not included.

## Conversion of plates, wires and particles

A plate (thickness 2 r0), cylinder or sphere (radius r0) of A reacting with a
matrix B through a diffusion-limited product layer converts the fraction alpha
of A, with z the product volume per volume of A used up and k_p the planar
constant of the same layer (van Loo 1990, eqs. 25-27):

    plate      (z alpha)^2 = 2 k_p t / r0^2
    cylinder   z/2 (1 + (z-1) alpha) ln(1 + (z-1) alpha)/(z-1) + z/2 (1 - alpha) ln(1 - alpha) = 2 k_p t / r0^2
    sphere     z^2/(z-1) - z/(z-1) (1 + (z-1) alpha)^(2/3) - z (1 - alpha)^(2/3) = 2 k_p t / r0^2

The left-hand sides follow from steady diffusion across the curved shell, which
the tests re-derive independently, so under an anneal k_p t becomes the time
integral of k_p. Inverting them gives the planar k_p from powder or wire
conversion data. Core and matrix are rigid; stresses and Kirkendall voids are
not modelled.

## Ordered B2 phases

Chemical order raises the activation energy of element k by (Helander and Agren
1999, eqs. 11-12)

    dQ_k^ord = sum_i sum_(j != i) dQ^order_kij [ y_i^a y_j^b - x_i x_j ],

with y^a, y^b the site fractions of the two B2 sublattices, x_i their mean and
vacancies a constituent. For a binary without vacancies this is Girifalco's
S^2 dQ^order/2 (their Appendix B). Site fractions and chemical potentials come from
pycalphad equilibrium of the ordered phase at each composition, so order relaxes
with composition, and dmu/dx is taken by central differences. With their Appendix A
parameters and pycalphad's Dupin Al-Ni description, the Ni tracer diffusivity
(their Fig. 7, 1000-1300 C) and the chemical diffusivity (Fig. 2) agree with their
curves within 0.04 decades for x_Al <= 0.5. Above 0.5, with structural vacancies,
the chemical diffusivity is 0.3 to 0.5 decades higher than theirs, both with the
Ansara (1997) (Al,Ni)(Ni,Va) model they used and with Dupin's (2001)
reformulation of it; their treatment of vacancies in the mobility is not spelled
out, and the module warns there.

## Cation diffusion in non-stoichiometric oxides

A metal ion moves by jumping into a vacant site of a sublattice it shares with
vacancies, so its flux follows the defect structure that the oxygen potential
sets (Hallstrom, Hoglund and Agren 2011, eqs. 3 and 13). For jump paths s with
a_s sites per a_O anion sites,

    J_M = -sum_s (a_s / a_O) y_Va^s y_M^s M_s (1 / V_m) dmu_M/dz
    D*_M = RT sum_s (a_s / a_O) y_Va^s y_M^s M_s / n_M                (eq. 7)
    D_M = (D*_M / RT) n_M dmu_M/dn_M                                 (eq. 9)

with n_M the metal per anion site. Magnetite has two paths, the normal
octahedral sites (vacancies at high p_O2) and the extra octahedral sites
(interstitial Fe at low p_O2), which gives the minimum of D* against p_O2;
hematite has one. Site fractions and potentials come from pycalphad at the
oxygen potential of O2 at the given partial pressure, and the derivative of eq. 9
is taken along that potential, which converges far better than differences in
composition over magnetite's narrow range. The shipped mobilities are those of
the paper (eqs. 10-11 and 14), fitted with Sundman's (1991) magnetite and
Kjellqvist et al.'s (2008) hematite. pycalphad's alfeo.tdb carries Kjellqvist et
al.'s Fe-O spinel and corundum term for term. Its spinel gives the defect fractions
of the paper's Fig. 3 within about 5 % from 1e-16 to 1e-2 atm at 1273 K, whereas
Sundman's own printed Table 1 has Fe+2:Fe+2:Va:O-2 = 7G - B where Kjellqvist has
7G + B, and with that sign misses Fig. 3 by a factor of three; the paper's "Sundman
data" evidently carried +B. The pressure of lowest D* then follows Fig. 4 within
0.25 decades from 1273 to 1623 K, but the level sits 0.1-0.25 decades below the
figure. That offset is the paper's: its eq. 7 with its eqs. 10-11 and its own Fig. 3
fractions lies 0.21-0.30 decades below its Fig. 4 at 1273 K, the same gap. Hematite
comes out 0.4-0.7 decades below the paper's Figs. 7-8 although the thermodynamics
is the one the paper used, a gap not yet explained.

## Oxide scale growth

Hallstrom et al. (2013) add oxygen to the same scheme for Cr2O3 on Cr. Cr jumps
into the empty interstitial sublattice of the corundum model
(Cr+2,Cr+3)2(Cr+3,Va)1(O-2)3 and O into its own vacancies, whose fraction they take
proportional to the interstitial Cr:

    J_O = -M_O (1 / V_m) dmu_O/dz,    M_O = k(T) y''_Cr,    D*_O = RT M_O      (eqs. 5-7)

In a fine-grained scale each mobility becomes f M_gb + (1 - f) M_b with
f = delta / grain size, delta = 5 Angstrom, and M_gb the bulk prefactor with alpha
times the bulk activation energy (eq. 8, alpha = 1/2). The paper solves the moving
boundaries numerically. For a scale that stays near stoichiometry, each flux is the
same at every depth, and integrating eq. (3) (through Gibbs-Duhem,
n_M dmu_M + dmu_O = 0) and eq. (5) across the scale with the interface balances of
eq. (15) gives X^2 = 2 k t with

    k = (1/n_g) int sum_s (a_s/a_O) y_Va^s y_M^s M_s / n_M dmu_O + int M_O dmu_O
      = (1/2) int (D*_M / n_g + D*_O) d ln p_O2,

from the metal/oxide equilibrium to the gas, with n_M local and n_g the
composition of the oxide formed at the gas side. This is Wagner's rate constant,
and V_m cancels. The neglected storage is of the order of the change of n_M
across the scale, under 2 % for Cr2O3. The integral is taken by Gauss-Legendre
over mu_O. With the Cr-O description of Kjellqvist et al. (2008), the paper's eqs.
(11)-(12), and 1 atm O2, the 24 h thicknesses come out at 345, 282, 658 and 588 nm
against the paper's 340, 280, 660 and 590 nm (625 C with 100 and 150 nm grains,
700 C with 200 and 250 nm). That closeness is partly a crossing. Their curves
(Figs. 8-9) start slower than any parabola and then grow faster, with a common
offset of about 5400 s, and their late-time k is 5 % above ours at 625 C and 8 %
above at 700 C. The oxygen tracer lines of their Fig. 7 agree within 0.07 decades.
Two things do not. Their Cr tracer lines (Fig. 6) lie 0.2-0.75 decades below eq.
(11) with the same thermodynamics, though their thicknesses follow eq. (11); since
the defect fraction cancels in D*_Cr / D*_O, their Figs. 6 and 7 are inconsistent
with eqs. (11)-(12) whatever the thermodynamics. And oxygen carries 18 % of k here
but about 10 % in their Fig. 11, a ratio fixed by eqs. (11)-(12) alone. The model assumes a compact,
adherent scale on a planar front, with no voids, cracks or evaporation.

## Homogenization

Each cell is in local equilibrium at its overall composition, so it has one set
of chemical potentials whatever phases it holds (Larsson and Hoglund 2009):

    J_k = -(1/V_S) Gamma*_k dmu_k/dz,       Gamma^phi_k = M^phi_k x^phi_k     (eqs. 2-5)

Gamma* is a Wiener bound (eqs. 6-7) or a Hashin-Shtrikman bound (eqs. 8-11) of
the phases' Gamma^phi weighted by their volume fractions. Local equilibrium is
either tabulated once per temperature with `pycalphad.equilibrium` and
interpolated linearly (binaries and ternaries), or, as in the paper's section
2.5, computed on demand: composition space is gridded, equilibrium is calculated
at a node only when a composition near it is first requested, and node results
are stored under the node's index. Between nodes the values are interpolated on
the Kuhn simplex of the grid cell, so potentials are continuous and dmu/dx is the
exact slope of the simplex. A four-element couple needs a few hundred nodes where
a complete table at the same step would need about 9000, and in a single-phase
quaternary couple the result approaches the ordinary solver as the step is
refined (2.1e-3 at 0.02, 4.8e-4 at 0.01, 1.4e-4 at 0.005). A step is accepted
when its update or its residual, in mole-fraction units, falls below the
tolerance; the residual test ends Newton cycling between two pieces of the
piecewise-linear potential. The paper's ideal flux term against oscillations
(section 2.4, eq. 13), J_k^ideal = -epsilon (Gamma*_k/u_k) RT du_k/dz, is an
option; it equals driving the flux with mu_k + epsilon RT ln u_k, which is how it
enters the Newton step. In an ideal solution it speeds diffusion by exactly
1 + epsilon, which the tests check. The two-phase couples tested here show no
oscillations without it, so there it only adds a little numerical diffusion. For anneals in binaries a
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

## Multicomponent precipitation

Svoboda, Fischer, Fratzl and Kozeschnik (2004, part I) derive the growth of a
precipitate in a multicomponent matrix from the thermodynamic extremum principle.
For a precipitate of fixed composition, infinite interface mobility and one molar
volume Omega for both phases, their eq. 45 reads

    drho/dt = (F - 2 gamma / rho) / (R T rho) [ sum_i (c_ki - c_0i)^2 / (c_0i D_0i) ]^-1
    F = (sum_i x_ki mu_0i - G_k) / Omega                                    (eq. 37)

with D_0i = RT M_i the tracer diffusivity of i in the matrix and G_k the molar
Gibbs energy of the precipitate at its composition, from pycalphad with its internal
degrees of freedom (ordering) at equilibrium. Nuclei take the composition of largest
driving force, the parallel tangent to the matrix: the precipitate is equilibrated at
MU_i = mu_0i + d for all but the last element and d is solved so the last potential
also shifts by d; the driving force is then -d per mole. The nucleation rate is part
I's eqs. 40-41 and 46-48, with beta* = 4 pi rho*^2 / (a^4 Omega) times the inverse of
the same sum and tau = 1 / (2 beta* Z^2), and Perez et al.'s corrected Zeldovich factor.
Each step's nuclei form a class at 1.05 rho* (part II, p. 158); classes are merged
when there are too many (number, volume, solute and Gibbs energy kept) and the most
populated class is split, 1 % apart in radius, when there are too few (part II
splits classes to keep their count). The split widens the distribution: against
the LSW rate the coarsening comes out 2.3, 1.2 and 1.0 times as fast for 20, 5 and
1 % splits.

Checks. For a dilute binary the growth law reduces to the LSW law with Perez et
al.'s eq. 11 rate constant, which the class model reproduces within 7 % over two
decades of time, with the LSW size distribution's 99th percentile at 1.33 <R>. For
Ni-5.2Al-14.2Cr at.% at 873 K with pycalphad's alcrni.tdb (Dupin et al. 2001, its
L1_2 checked against the paper's relations 12-28) the gamma/gamma' equilibrium and the
critical nucleus (16.37 Al, 8.25 Cr at.%) agree with Thermo-Calc on the same
database (Sudbrack et al. 2007, Table 3) within 0.1 at.%. Against the atom-probe
kinetics of Sudbrack et al. (2006), with Campbell's mobilities and gamma = 22.5 mJ/m^2
from the measured coarsening and nothing adjusted, the state at 4 h matches (5.3 %
against 5.2 %, 3.2e24 against 3.2e24 m^-3, 1.54 against 1.27 nm). Coarsening, from
<R>^3 over 64-1024 h, is 4.4-5.0 times faster than measured with step limits 1 to 8
times tighter, close to the paper's own estimate with these mobilities, which they
attribute to diffusion through the gamma' as well.

Two class-model details decide whether the late stage is right. Nuclei are held back
until they make up a millionth of the population: late in a run the nucleation rate
is tiny but not zero, and a class for each step's 1e-100 nuclei fills the array,
whose merging then collapses the real particles into one class that stops
coarsening (it did, for hundreds of hours, before this). And shrinking classes may
dissolve within a step only while their solute content together, not each alone,
stays under the matrix step limit; otherwise a population that all shrinks at once,
as on heating, vanishes in a single step.
Nucleation comes too early: classical theory at a 5 kT barrier gives 2.6e25 m^-3 s^-1
against the measured 5.9e21, a comparison the authors themselves warn against. The
final volume fraction is bounded by Dupin's 11.8 % rather than the measured 15.6 %.

With tracer diffusivities in the precipitate, class compositions evolve by part I
eqs. 27-39: per class a linear system in (drho/dt, dc_i/dt), symmetric and positive
definite as the dissipation is. The printed eq. 32 carries 5/(3 c_01 D_01) where
eq. 25, eq. 33 and the dissipation itself give 5/(c_01 D_01); the latter is used, and
a test checks the solved rates against eqs. 27-28 by differentiating the dissipation
numerically. The precipitate's chemical potentials at each class composition come
from a constrained Newton minimisation of its Gibbs energy over site fractions,
which matches pycalphad within 0.01 J/mol where gamma' is one phase. In
Ni-5.2Al-14.2Cr the gamma' then leaves its nucleus composition for Dupin's solvus: Al
falls as measured, Cr rises where the measurement falls, Dupin putting the gamma' Cr
at 9.5 % against the measured 6.8 %. A second alloy, Ni-7.5Al-8.5Cr (Booth-Morrison et
al. 2008), matches <R> at 1 h. With evolving compositions the coarsening rate is
5.2-5.8 times the measured one for that alloy and 5.6-6.5 times for Ni-5.2Al-14.2Cr,
over a fourfold range of step limits and 100-800 classes, so the gap follows the
mobilities rather than the alloy; the paper's own estimate for isolated particles is
6.4 times.

Misfit. Part I eq. 37 subtracts the misfit energy lambda per volume of precipitate
from the driving force. Appendix A gives it as lambda0 = eps0^2 E / (1 - nu) for a
linear misfit eps0, or lambda0 f(S), f = 2 S (1 - ln S) - S^2, when the matrix yields
with S = sigma_f (1 - nu) / (E eps0); f reaches 1 at S = 0.199 and peaks at 1.456 at
S = 0.567, both checked.

Grain junctions. Clemm and Fisher (1955) give the coefficients a, b, c of a nucleus
bounded by spherical caps of radius r at a grain boundary (a lens), an edge where
three grains meet or a corner where four meet: it removes a r^2 of grain boundary,
creates b r^2 of interface and holds c r^3. With k = gamma_gb / (2 gamma) the barrier
is (b - 2 k a)^3 / (36 pi c^2) times the homogeneous one at the same critical radius
of curvature; it vanishes at k = 1, sqrt(3)/2 and sqrt(2/3). Their Fig. 6 barriers
for ferrite in austenite are reproduced within 0.5 % for the bulk, boundary and edge,
but the corner value printed, 0.0096e-8 erg, is 13 % below what their own
coefficients give (0.0109e-8), although those pass every internal check (the barrier
ratio equals the volume ratio at every k, and the limits hold); the figure is taken
as a computing slip. Only the barrier changes; the attachment rate and Zeldovich
factor keep their homogeneous form, and a junction nucleus enters the classes as a
sphere of the critical radius of curvature, larger in volume than the lens it stands
for. These two choices are this package's.

Temperature schedules. The temperature may follow a piecewise-linear schedule; steps
then also move T by at most 1 K, and the class Gibbs energies and the nucleus are
recomputed at each temperature. A run along an 873 K to 848 K or 893 K step agrees
with two chained isothermal runs within 2 % in volume fraction, number and radius; at
893 K most of the gamma' dissolves, the small particles falling below the new
critical radius.

## Microsegregation and homogenization

For a constant partition ratio k the closed forms are taken as Kobayashi (1988)
states them: Scheil k (1 - f_s)^(k-1), the lever rule k / (1 - (1 - k) f_s), and
Brody-Flemings, Clyne-Kurz and Ohnaka, with gamma = 2 D t_f / L^2 for parabolic
growth over half an arm spacing L. Kobayashi's exact solution of the Brody-Flemings
model (his eqs. 37-40, confluent hypergeometric series) converges slowly at f_s = 1:
for P in delta-Fe (k = 0.13, gamma = 0.8) it reaches 4.5130 by 5000 terms, where
the paper prints 4.48, the value of a series stopped near 1000 terms. Clyne-Kurz
and Ohnaka give about 1.8 there, as he says.

The same model is solved numerically on the package's moving-grid scheme, in a plate,
cylinder or sphere: the solid grows from the spine at a prescribed fraction solid,
holds k C_L at the interface and diffuses with constant D, while the liquid stays
uniform. The solid's response is linear in C_L, so two solves per step give the
liquid composition that closes the solute balance exactly (to 1e-15). Against the
exact solution the interface composition is within 0.2 % up to f_s = 0.95 and 4.49 at
f_s = 1, converging with the grid; with no solid diffusion it gives Scheil and with
fast diffusion the lever rule, in every geometry.

For any number of elements the Scheil-Gulliver path follows the algorithm Bocklund
et al. (2020) give: equilibrate the remaining liquid at each temperature, set the
solid aside, shrink the liquid. On an ideal binary built so that x_s / x_l = k exactly
it reproduces the Scheil equation within 1.5 % at 4 K steps, the error falling with the
step. A Scheil or back-diffusion record is laid across the volume element core first
(fraction f at r = L f^(1/(p+1))) as the starting profile of a homogenization anneal,
whose progress is followed by the residual segregation index (max - min over its
start). In a closed plate, cylinder and sphere the slowest mode decays as
exp(-alpha^2 D t / L^2) with alpha = pi, 3.8317 and 4.4934, which the anneal follows
within 1 %. The worked example takes Ni-5.2Al-14.2Cr through Scheil solidification
on Dupin's description and a 1200 C anneal of a 25 um dendrite arm with Campbell's
mobilities.

