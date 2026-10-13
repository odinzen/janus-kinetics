# Theory

The equations janus-kinetics implements, with the paper and equation number
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

With a tolerance the step adapts. Backward Euler errs by about (h^2 / 2) d^2x/dt^2 per
step; the second derivative is estimated from the change in dx/dt between consecutive
steps, its largest value over the cells sets the error, and a step over the tolerance
is halved and redone, as is one whose Picard iteration fails. Steps are dt / 2^k, so a
constant D needs few factorizations. This pays when profiles are wanted over decades of
time: for a couple saved at 10 s to 1e5 s, 107 adaptive steps reach a worst error of
1.5e-3 against a time-converged run, where 400 equal steps leave 8.9e-3. Along a
1e4-fold rise of D the gain is about 1.5 times fewer steps. For a single anneal judged
at its end equal steps do as well or better, since early errors decay; the largest
cell's error is used rather than a mean, which is as efficient and does not change
meaning with the length of untouched material in the domain.

## Cylinders and spheres

The planar solver, the layered moving-interface solver and the homogenization solver
also run in a long cylinder or a sphere, the one-variable geometries of DICTRA
(Borgenstam et al. 2000): dx/dt = r^-p d/dr (r^p D dx/dr) with p = 1 or 2. Cells hold
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

Fisher's equations drop diffusion along the boundary direction inside the grains.
Whipple (1954) solved the problem with it kept; it is used here as Le Claire (1963)
writes it, a single integral for c/c_0 in eta = y/(Dt)^1/2, xi and beta = (D'/D)(delta/2)/
(Dt)^1/2 (his eq. 7), and the layer average of a sectioning experiment (eq. 13). Le
Claire's reading of such data is ln c_bar linear in y^6/5 with D' delta = (4D/t)^1/2
(0.78 / |slope|)^5/3 = 1.322 (D/t)^1/2 |slope|^-5/3 (eq. 15), valid for beta above about 10.
Whipple's own paper has not been read; his solution is taken from Le Claire.

Checks. Above beta of 10 the boundary concentration depends on eta beta^-1/2 alone, as
Le Claire's Fig. 5 shows; the exact solution of Fisher's equations approaches it with
growing t1 (2.3 % apart at t1 = 1600), being its large-beta limit; the slope of eq. 13
approaches the asymptotic eq. 14 with depth and its 6/5 slope stays within 0.75-0.81.
Read with eq. 15, an exact Whipple profile at beta = 100 returns D' delta 3.4 % high
(its own coefficient being 0.765), while Fisher's linear reading returns a third of it,
the 'about three times too small' Le Claire reports.

For a diffusant deposited as a thin layer (Suzuoka 1961) the layer average is Whipple's
differentiated in eta (Le Claire p. 356). Its 6/5 slope, fitted over eta beta^-1/2 = 2-10,
is 0.738 and 0.744 at beta = 10 and 100, against Le Claire's 0.72 beta^0.008 (eq. 16), and
read with that coefficient eq. 15 returns D' delta within 2 %. Near the surface the
boundary term is negative, the boundary drawing material from the lattice, as Suzuoka
notes; at beta = 1 the profile still rises at the start of the usual fitting range.

Which picture applies follows Harrison (1961): type A once lattice diffusion crosses a
grain within the anneal, a_g^2 / D << t (his eq. 1), when the solid diffuses as a whole
with Hart's average; type C while lattice diffusion stays within the boundary layer,
(D t)^1/2 << delta (eq. 5); type B, Whipple's, between. Harrison leaves '<<' unquantified;
the package's harrison_regime takes it as a factor of 10, which can be changed.

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

Layered stacks take adaptive steps too. Compositions count in mole fraction and
each interface position in thicknesses of the thinner layer beside it, so a
nanometre reaction layer is followed as closely as a thick one. This matters: a
200 nm beta layer (D = 1e-17 m^2/s) between faster alpha and gamma thins to 1.565 nm
at 15 s and then grows again (0.1 s and 0.02 s steps agree), but steps of 0.5 s or
more carry its interface past the next one and report it consumed. A layer that
seems to vanish within an adaptive step while it still holds 2 % of its first
thickness is retried with a shorter step; a real end, such as a 0.5 um film eaten
by a growing layer (near 84 s with 0.2 s steps and adaptively), still stops the run.

Graded grids. The interface takes each side's flux from the gradient over the half cell beside it,
first-order accurate in that cell's width, so the cells there decide the accuracy. A layer may
place its faces at any fractions of its width (kept as it stretches); `graded_faces` makes them
grow geometrically away from one or both ends, the widest `ratio` times the narrowest. On Zener's
growing sphere 300 matrix cells graded 100:1 give the radius at 1e5 s within 0.17 %, against 0.35 %
with 1200 uniform cells (cylinder 0.11 against 0.24 %), and the error falls in proportion to the
interface cell; the two-sided similarity couple on 50 + 100 graded cells lands within 0.15 %,
closer than 400 + 800 uniform ones, and the ternary local-equilibrium couple on 40 graded cells a
side matches a 400-cell uniform run within 0.4 % in position and 1e-4 in the operating tie-line,
where 40 uniform cells miss by 10 %. A uniform phase stays uniform on a stretching graded grid and
every element is conserved to round-off.

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

Several layers around a wire or particle. Van Loo treats one layer; for several
the planar line-compound model is carried into concentric shells here. In a layer
of fixed composition each element is conserved, so r^p (c_i w + J_i) is the same at
every radius (p = 1 cylinder, 2 sphere). With the Darken fluxes J_A = x_A x_B M_A
dS/dr / V and J_B = -x_A x_B M_B dS/dr / V of the planar model, the lattice velocity
w and dS/dr both fall as r^-p, and integrating dS/dr across the layer gives

    r^p J_i = (planar flux of the layer) with d replaced by the integral of r^-p dr:
    ln(r_out / r_in) for a cylinder, 1/r_in - 1/r_out for a sphere.

Multiplied by r^p, every interface balance c_R (v - w_R) - c_L (v - w_L) = J_R - J_L
is the planar one in V = r^p v and W = r^p w, so the planar velocities apply as they
stand. The core passes no flux and its lattice velocity W / r^p must stay finite at
the centre, so the core's lattice is the frame. Its volume r^(p+1) is followed
rather than its radius, whose rate diverges as it vanishes; it counts as used up at
1e-4 of its radius, 2.4e-8 (sphere) and 1.4e-7 (cylinder) of the conversion time
early. With one element diffusing the Kirkendall plane sits on an interface, and in a
cylinder or sphere a marker a rounding error off it would drift away, since lattice
speeds fall as r^-p; such a marker is held on the interface.

Checks. One layer reproduces eqs. 25-26 within 1e-9 for both shapes and with A, B or
both diffusing. Two layers in a sphere agree with the layered-stack solver whose
phases have a composition range delta and D = D_int / delta: 0.64 and 1.9 nm at 200
and 1000 s for delta = 0.02, half that for 0.01, against 130 and 400 nm of interface
movement. A 1 m core grows its layers as a plate within 2e-7, and with a core and
then the next layer used up inside a finite shell every atom is kept within 4e-8. A
marker in the outer shell keeps the volume between it and the shell's outer surface,
both riding the same lattice, to 1e-9.

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

## Linear-parabolic oxidation

Deal and Grove (1965) take the oxidant across the gas boundary layer, through the
oxide by diffusion and into reaction at the oxide/silicon interface, with equal fluxes
in steady state. The thickness then grows as eq. 9, dx/dt = B / (A + 2x), with
A = 2 D_eff (1/k + 1/h) and B = 2 D_eff C*/N_1 (eqs. 12a-b), so x^2 + A x = B (t + tau)
with tau = (x_i^2 + A x_i)/B for an initial oxide x_i (eqs. 11-13). Eq. 13 is evaluated
as 2 B s / (A + sqrt(A^2 + 4 B s)), s = t + tau, which keeps its precision for thin
oxides; eq. 9 itself is integrated along temperature schedules. A and B are fitted as
the paper does, x_0 against (t + tau)/x_0 (slope B, intercept -A).

Checks. The B/A columns of Tables I and II are B over A. Arrhenius fits of the tables
give 29.7 and 16.3 kcal/mol for B and 46.6 and 44.6 for B/A (dry and wet) against the
28.5, 16.3, 46.0 and 45.3 of Figs. 9 and 11. The lines of Fig. 4, fitted to their own
pixels, reproduce Table I within 4 % in B and 0.005 um in A at 1100, 1000 and 920 C.
Table II's tau differs from eq. 12c with its own A and B by up to 27 %; the text says
tau was estimated graphically, and the package uses eq. 12c. Their oxide Debye
lengths (about 150 and 6 A) follow from their C* values.

## Competing compound layers in thin films

Gosele and Tu (1982) explain why thin-film couples often grow one compound where the
phase diagram has several. Each compound layer's A flux is limited by diffusion and by
the reactions at its interfaces, j = Delta C kappa_eff / (1 + x kappa_eff / D) with
1/kappa_eff = 1/kappa_1 + 1/kappa_2 (eqs. 6-7), so a layer grows linearly below its
changeover thickness x* = D/kappa_eff and parabolically above it (eqs. 11-13), which is
Deal and Grove's law with A = 2D/kappa and B = 2 G Delta C D. Two layers between saturated
phases gain from their own flux and lose to their neighbour's (eqs. 15-16, composition
factors eqs. 19-21), and both grow only while j_beta/j_gamma lies between r_1 and r_2
(eqs. 24-29). A layer of zero thickness that would shrink stays absent, so the second
compound appears only once the first passes a critical thickness (eq. 34). Eq. 34 takes
the first layer as diffusion controlled; with the second at zero thickness the exact
onset is x_crit - x*_gamma, which is what the solver finds. (Eq. 33 prints a stray T
before D_gamma that eq. 34, derived from it, does not carry.)

Checks. r_1 and r_2 equal the ratios of the composition factors (2/3 and 4/3 for AB and
A2B between pure A and B). One layer follows Deal and Grove's law to 1e-6; the second
layer appears between first-layer thicknesses bracketing x_crit - x*_gamma; thick layers
settle on the diffusion-controlled thickness ratio within 0.6 %, of order x*/x; and
interface-controlled layers grow linearly at the rates of eq. 32, or one shrinks away
when the flux ratio leaves the window.


## Nucleation at an interface

d'Heurle (1988) explains why some compounds (NiSi2, IrSi3, Rh4Si5) appear suddenly at a sharp
temperature instead of growing by diffusion: a new phase AB between A and B replaces one interface
by two, raising the interfacial energy by Delta sigma = sigma_2 + sigma_3 - sigma_1 (eq. 11), and when
it forms from a neighbouring compound its driving force is small. A layer pays for itself beyond the
specific thickness d_s = Delta sigma / Delta h (eq. 12). A discrete nucleus is a lens of two caps on the
old interface; with equal energies it adds (5 pi / 4) R^2 and holds (5 pi / 12) R^3, so R* = 2 sigma /
Delta h (eqs. 14-16). For unequal energies the rim follows Neumann's triangle; the lens exists while
each energy is below the sum of the other two (eqs. 17-18), a cap is a hemisphere when sigma_1^2 +
sigma_2^2 = sigma_3^2 (eq. 19), each cap's critical radius is 2 sigma_i / Delta g and the barrier is a
third of the interface energy added. Checks: Table II's specific thicknesses; the molar volumes from
his unit cells (Ni2Si 19.6, Ni3Si2 34.4, NiSi2 23.8 cm^3/mol) and the NiSi2 driving force of Table III,
55 to 99 cal/cm^3 between 300 and 1100 K; the equal-energy lens; eq. (19); and, with sigma_2 = sigma_3,
the grain-boundary lens of Clemm and Fisher coded separately for precipitation, at every k. Two of his
worked numbers are slightly off his own inputs: d_s for Ni2Si comes to 0.137 nm (printed 0.15), and the
NiSi driving force to 300 cal/cm^3 (printed 285).
## Electromigration stress

In a metal line held by a rigid passivation the electron wind moves atoms along the
line, and where that flux diverges it builds stress. With vacancies in equilibrium,
Korhonen, Borgesen, Tu and Li (1993, eq. 8) reduce this to a diffusion equation for the
stress,

    d sigma/dt = d/dx [ kappa (d sigma/dx + G) ],   kappa = D_a B Omega / kT,   G = rho j |Z*| e / Omega,

with x along the electron flow. It is solved by backward Euler on a uniform grid, with
ends that are blocked (no flux), voids (their eq. 14, d sigma/dx + G = sigma / w, mirrored
at a right end) or reservoirs at a fixed stress, kappa constant or a function of stress
(eq. 18), and optionally ideal plasticity: a yielded zone holds the yield stress and
exchanges no flux with the rest of the line, so its edge advances as the elastic stress
beside it reaches yield (section III, eq. 11). The steady state of a line blocked at both
ends, sigma = G L (1/2 - x/L) (eq. 12), has the end-to-end difference G L that Blech's
threshold (1976, eq. 6) sets equal to the largest sustainable Delta sigma:
(jL)_c = Omega Delta sigma / (|Z*| e rho), eq. (6) as Blech's erratum (1977) corrects it (the paper printed eq. (5)
without rho and eq. (6) inverted on the left); his own 1.2e-9 cm^2/dyn for Z*/Delta sigma follows from the corrected form.

Checks. The solver converges at first order in the step onto eq. 9 (semi-infinite line),
eq. 13 (blocked at both ends) and eq. 15 (a void at one end, alpha = 5), and settles on
eq. 12 to 1e-9. Eq. 15 prints its leading term as -G xi; only -G L xi starts the line
stress-free and leaves the steady state the paper describes. Under yielding the front
advances at G kappa / sigma_y within 0.3 % and the stress ahead of it follows eq. 11 within
1 %. Their Fig. 9 fit (G = 16.5 MPa/um, kappa = 2.6 um^2/h, 270 h), digitized, matches eq. 9
within 0.006 in log10 stress. The diffusivity they infer from that kappa, D_a = 3e-16 m^2/s,
is 50 times what kappa = D_a B Omega / kT gives with their B = 50 GPa (6e-18 m^2/s); it
follows from converting hours as if they were minutes. Blech's own numbers agree with his
threshold: 1.1e5 A/cm^2 in a 115 um stripe is the 1260 A/cm he quotes.

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

Steps adapt as in the planar solver when a tolerance is given, and a step whose
Newton iteration fails or leaves the table is halved and redone. For an alpha/beta
couple saved at 0.1 s to 200 s, tolerance 1e-4 takes 217 steps for a worst error of
9e-4 against a 0.01 s-step run, where equal steps need about 800.

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

Site counts. Cahn (1956, eq. 1) gives the grain boundary area, edge length and corner
count per volume of equal tetrakaidecahedral grains whose square faces lie D apart:
3.35 / D, 8.5 / D^2 and 12.0 / D^3. The exact values of that solid, (3 + 6 sqrt 3) / 4D,
6 sqrt 2 / D^2 and 12 / D^3, are used, and a test measures the same cell as the
Voronoi cell of a bcc lattice. Part II (p. 162) counts the nucleation sites on
dislocations as the atoms on the line; the same count is applied here to grain
junctions, with the atomic distance w = (Omega / N_A)^(1/3): S / w^2 atoms in one
boundary layer, L / w along the edges and one site per corner. That convention is this
package's; neither paper fixes it for junctions.

Temperature schedules. The temperature may follow a piecewise-linear schedule; steps
then also move T by at most 1 K, and the class Gibbs energies and the nucleus are
recomputed at each temperature. A run along an 873 K to 848 K or 893 K step agrees
with two chained isothermal runs within 2 % in volume fraction, number and radius; at
893 K most of the gamma' dissolves, the small particles falling below the new
critical radius.

Competing populations. Part I's mass balance (eq. 4) sums over precipitates of every
kind, and part II (p. 158, section 3.5) gives each phase, and each nucleation site of a
phase, its own array of classes to follow precipitation sequences. Here each such
population nucleates, grows and coarsens as above while the matrix holds what all of
them leave; they share one time step, the matrix step limit counts their depletion
together and shrinking classes of all of them draw on one dissolution budget. Part I's
single molar volume (p. 167) is required of every population, so that volume fractions
carry the mass balance. Checks: particles held as one population or dealt into two give
the same run within 1e-9, and an idle second population changes nothing to the bit.
Seeds of gamma' and bcc in Ni-8Al-30Cr at 1073 K, in Dupin's three-phase field, grow
until each meets its own Gibbs-Thomson condition, F = 2 gamma / rho, within 1e-3, and
hold the lever-rule fractions less the capillary shift (22.13 against 22.26 % gamma',
6.71 against 6.77 % bcc).

That check exposed a step-control fault shared with single-phase runs. Near a
stationary state a step longer than the matrix's relaxation time overshoots, and the
matrix swung about the fixed point by up to its step limit, 2e-4 in mole fraction,
more than the capillary supersaturation of 80 nm particles: F rho / (2 gamma) wandered
between 0.72 and 1.09. A step is now halved whenever the matrix rate reverses between
steps by more than a thousandth of the step limit (smaller reversals are rounding noise
at the fixed point).

## Growth of one particle with local equilibrium

Chen, Jeppsson and Agren (2008) recast the exact growth of a sphere in a binary, v = 2 D lambda^2 / R
with 2 lambda^2 - 2 lambda^3 sqrt(pi) exp(lambda^2) erfc(lambda) = Omega (eqs. 1-2), as a steady-state
flux over an effective distance n R, n = Omega / (2 lambda^2) (eqs. 4-6), and give every element of a
multicomponent matrix its own n_j from its own supersaturation Omega_j. The growth rate and both
interface compositions then follow from local equilibrium with the Gibbs-Thomson pressure on the
particle (eq. 9) and one flux balance per independent element, in diffusivities (eq. 10) or in
mobilities and chemical potentials (eq. 13).

Eq. (2) is Zener's eq. (12) for a sphere with alpha = 2 lambda; near Omega = 1 it cancels in floating
point, so there 1 - f is taken from the asymptotic series of erfc, checked against Laplace's continued
fraction to 1e-9. With a diagonal D and no capillarity each element's field is the exact binary one,
so eq. (10) is exact for growth from zero size: a sphere grown by the sharp-interface solver in an
ideal ternary, one solute diffusing 4 times faster than the other, follows it within 0.1 % in
d(R^2)/dt and 2e-6 in the operating tie-line. Eq. (13) writes the flux with interface quantities, so in
an ideal dilute matrix it falls short of the exact rate by x^I ln(x^M/x^I) / (x^M - x^I) (0.53 for a
threefold supersaturation); the paper prefers it for the cross terms it carries through the chemical
potentials. Where Omega_j falls outside (0, 1), eq. (2) has no root and n_j = 1 is used, a package
choice. The paper's Fe-Cr-C example needs the commercial TCFE4 and MOB2 databases and is not
reproduced.

As a growth law for the precipitation solver. A class keeps its nucleation composition and Gibbs
energy, so local equilibrium is one condition, sum_i x^P_i mu^I_i = G^P + V_m (2 sigma / R + misfit),
with the flux balances; all classes are solved together by Newton, each starting from its previous
step. A class too small for any matrix composition to balance its capillary pressure (a few tenths of
a nanometre) has no local equilibrium; it is dissolving and shrinks at the extremal rate of eq. (45).
Linearized about the tie-line, eq. (13) gives dR/dt = (F - 2 sigma V_m / R) / (R Q) with
Q = sum_j a_j dx_j / (x_j M_j) over the solutes, a_j = x^P_j - x^P_solvent x_j / x_solvent, so Philippe
and Voorhees' K = 8 sigma V_m / (9 Q) follows; a seeded run in an ideal ternary coarsens at it within
3 %. The extremal law also charges the solvent's flux, sum over all elements; with a solvent 100 times
slower than the slowest solute its K is 1.8 times lower, so the two laws part where the solvent's
diffusion matters. For gamma' in Ni-5.2Al-14.2Cr at 873 K (the Sudbrack case of the precipitation
section) local-equilibrium growth gives 1.75 nm, 2.5e24 m^-3 and 5.8 % at 4 h (extremal law 1.54 nm,
3.2e24, 5.3 %; measured 1.27 nm, 3.2e24, 5.2 %) and K = 4.2e-31 m^3/s between 256 and 1024 h, 4.8 times
the measurement against 4.5 for the extremal law: neither closes the coarsening gap, which Sudbrack
et al. put down to diffusion through the gamma'. The two forms agree within a few percent there. It runs
about seven times slower than the extremal law.

## Precipitation along heat treatments

Myhr and Grong (2000) follow a size distribution through ageing, up-quenching and heating
ramps with deliberately simple physics: an empirical nucleation rate
j = j0 exp[-(A0/RT)^3 / ln(C/C_e)^2] exp(-Q_d/RT) (eq. 3), the dilute growth law
dr/dt = (C - C_i) / (C_p - C_i) D / r with C_i = C_e exp(2 sigma V_m / rRT) (eqs. 4-5), an Arrhenius
solvus (eq. 17) and the solute balance (eq. 11). The distribution moves through fixed classes
by their implicit upwind control-volume scheme (eqs. 12-16, Table 1), with wall velocities from
the previous step's matrix content and nuclei entering at 1.05 r*. Their Table 2 holds every
input for an AA 6082 alloy (`aa6082()`).

Two points the paper leaves open are this package's. Eq. (4) changes sign once C_i passes C_p,
below 2 sigma V_m / (RT ln(C_p/C_e)) (0.43 nm at 180 C), so such particles dissolve at once. And
the time step: the paper only says it must keep C from changing much; here it is also kept
below one class width of travel for growing classes, since the implicit scheme, stable at any
step, otherwise smears the distribution over many classes (89 steps to 1e4 s at 180 C spread the
particles to twice their size). Shrinking classes are left out of that limit: they move fastest
just above the size where eq. (4) fails and would set the step for the whole run, and leaving
them out changes the results by 0.3 % at most.

Checks against all five simulation figures. At the start of ageing Table 2 gives a barrier of
2.97 kT ('about 3'), j = 5.2e18 (Fig. 3a: 4.9e18) and r* = 0.81 nm. Ageing at 180 C follows Fig. 3
within 3 % in mean radius and 5 % in number density to 3e6 s (the last number density, at
8.5e6 s, moves -3 to +3 % with the class width); f = 0.008 at 1e4 s, the paper's peak-aged state.
An up-quench to 380 C follows Fig. 5 within 1.5 % in f and 2 % in radius, dissolution ending at
0.32 s where the paper draws it (0.317 s). The distributions of Fig. 6 are built from the text
(mean 38 A, f = 0.008): their heights match the drawing within 4 %, but every drawn curve sits
about 10 % lower in radius, so the drawing, not the runs of Fig. 7 (which start at 38 A), is taken
to be off. At 480 C (Fig. 7) the LSW and b = 0.4 starts dissolve, the broader ones regrow to the
equilibrium fraction with 1.38e-7 m particles at 1000 s, and the b = 0.6 number density levels at
1.5e17 against 1.65e17. Heating at 1 C/s (Fig. 8) reproduces the three tempers' starting radii,
their meeting near 300 C and the 3.65e-8 m maximum near 488 C. With the interfacial energy changed
for the heating (Fig. 9) every sigma = 0.05 marker is matched within 0.0003 in f and the readable
sigma = 0.8 ones within 0.00012; only the regrowth after the sigma = 0.8 dip runs about 8 C behind
the figure, unchanged by refining steps or classes.

## Late-stage coarsening in closed form

Philippe and Voorhees (2013) linearize local equilibrium about the tie-line dC = C^beta - C^alpha
and find a particle's growth rate (eq. 28)

    dR/dt = [dC^T G dC^inf - 2 sigma V_m / R] / (R dC^T M^-1 dC)

with G the Hessian of the matrix Gibbs energy and M the mobility matrix, D = M G. Its long-time
solution is the LSW one with new amplitudes: <R>^3 = K t, K = 8 sigma V_m / (9 dC^T M^-1 dC)
(eq. 42); a matrix supersaturation along the tie-line, decaying as t^(-1/3) with the amplitude of
eq. (43); N_v = 0.21 phi_eq / (K t) (eq. 44; the factor is 3 / (4 pi <u^3>) of the LSW distribution,
0.2113); and a precipitate composition that drifts off the tie-line by G_beta^-1 G_alpha (eq. 45). The
functions take dC, M, G and the precipitate's Hessian over the independent components.

Checks. In the binary limit eq. (42) is Perez et al.'s LSW constant exactly. Eq. (45) is rebuilt from
eqs. (35), (41) and (43): the mean particle has the critical radius, so its matrix interface sits at
the far-field composition, and eq. (35) carries that to the precipitate side. The precipitation solver
follows these laws in an ideal ternary with a line-compound precipitate, two solutes diffusing five
times apart and particles seeded at the equilibrium composition: SFFK's dissipation sum over all
components, sum_i dx_i^2 RT / (x_i D_i), is dC^T M^-1 dC for a matrix M^-1 with a solvent term in every
entry. From 1e7 s on, d<R>^3/dt averages K within 2 %, d(1/N_v)/dt averages K / (0.2113 phi) within 2 %,
the supersaturation stays on the tie-line, its critical radius closes on <R> (0.935 to 0.997), and its
size is eq. (43)'s within 2 % from 3e8 s. With the two solutes equally mobile K would be 2.6 times
larger, and keeping only the slow one (eq. 48) would overstate it by 31 %.

## Interfacial energy from broken bonds

Sonderegger and Kozeschnik (2009) estimate the energy of a coherent interface from the
enthalpy of solution, extending the nearest-neighbour broken-bond count of Becker and
Turnbull to every neighbour shell and to any number of components. A bond to the k-th
shell weighs f = (r_k / r_1)^-m (eq. 4), and

    gamma = n_S z_S,eff / (N z_L,eff) Delta E_sol,   n_S = n_V^(2/3)          (eqs. 9-10)

with z_L,eff the weighted coordination (eq. 7) and z_S,eff the weighted bonds cut per
interface atom (eqs. 8, 11). Shells past the k-th are smeared into a continuum (Appendix
II), adding 4 pi n_V r_1^m r_k^(3-m) / (m - 3) and pi n_V^(4/3) r_1^m r_k^(4-m) / (m - 4);
both diverge for m <= 4, as the paper notes.

The paper counts bonds in periodic volume elements; here they are lattice sums. In a
Bravais lattice every lattice vector v projects onto the interface normal n as a whole
number of interplanar spacings, so the bonds of type v cut per unit area are exactly
n_V |v . n| / 2, and z_S,eff = n_V^(1/3) / 2 sum_v |v . n| f. Averaged over orientations
|v . n| becomes |v| / 2, which gives the mean in closed form (it agrees with an average
over 4000 random orientations to 0.03 %).

Checks. With 100 shells all 132 entries of Table II (22 orientations, fcc and bcc,
m = 6 to 8) are reproduced within 0.1 % except one: [4,4,3] bcc at r^-8, printed 0.3297
against our 0.2969. Every other row falls from r^-7 to r^-8, and the printed value
would rise above its own r^-7 entry, so it is taken as a misprint. The first-neighbour
limits 0.2291 (fcc [111]) and 0.2227 (bcc [110]) of p. 506 are Turnbull's count exactly,
and first neighbours alone underestimate z_L,eff by 20 % and 53 % (p. 504). Among
orientations with indices up to 10 the fcc extremes are those of Table III ([111],
0.2871, and [10,5,1], 0.3398); in bcc the minimum ([110]) and the 17.6 % anisotropy
agree, but the largest value is at [6,1,1], 0.3418, 0.1 % above the [10,2,1] the paper
names. Two quoted numbers are slightly off the converged sums: z_L,eff = 14.442 (fcc)
and 12.234 (bcc) on p. 503, against 14.4539 and 12.2537 here, and the r^-7 and r^-8
orientation means of Table II, which differ from the closed form by up to 0.7 % with
alternating sign; the r^-6 means of Table III agree within 0.2 %.

Enthalpy of solution. Appendix I derives the multicomponent bond energy from dissolving
one mole of precipitate in an infinite matrix, so Delta E_sol = sum_j x_j^P H_j^M - H^P
with H_j^M the partial molar enthalpies of the matrix. It is computed from
Delta G = sum_j x_j^P mu_j^M - G^P by Gibbs-Helmholtz at fixed compositions, each phase
held homogeneous with its ordering relaxed (the precipitation module's phase
minimiser). Holding it homogeneous matters: at the gamma solvus pycalphad's equilibrium
at fixed composition splits an ordered L1_2 description of the matrix into
gamma + gamma' half a kelvin below T, and the derivative would follow the tie-line.
A regular solution with temperature-dependent parameters gives the exact partial
enthalpy to 1e-5. For gamma' in binary Ni-Al between 773 and 973 K, Dupin et al.'s
description gives 40-45 mJ/m^2 with the r^-6 mean factor, inside the 14-80 mJ/m^2
measured range of Table VI; the paper's own 36.2 used a commercial database.

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
mobilities. Purdy and Kirkaldy (1971)
review the same picture: a sinusoid decays as exp(-pi^2 D t / l^2) (eq. 6), a ternary one in two modes at the
eigenvalues of D (eqs. 11-16), and a fast solute ends slaved to a slow one with amplitude -(D_12 / D_11) A_2
(eq. 21); the planar solver follows all three, eq. 21 within D_22 / D_11 as its derivation allows.

Back-diffusion with CALPHAD tie-lines. `solve_solidification` is the volume-element
calculation of DICTRA's solidification runs for any number of elements: the primary solid
grows from the arm's centre as the temperature falls, diffusing with its own
interdiffusion matrix on a grid graded toward the interface; the liquid beyond is kept
uniform; the interface follows the tie-line of local equilibrium at the current
temperature, and the element is closed. It is the layered-stack solver with a liquid layer
of very high diffusivity, so the interface balance, conservation and geometry are those
already checked. Its two limits come out on an ideal A-B-C alloy with a 35 K equilibrium
freezing range (partition ratios about 0.36 and 0.57): with the solid frozen the fraction
solid and the liquid follow pycalphad's Scheil-Gulliver path within 1e-3 down to 1560 K, in a
plate and around a cylinder alike; with fast diffusion in the solid the fraction solid is
the lever rule's within 3e-3 and the liquid runs out at the equilibrium solidus. In between
it lies between them and rises with the solid diffusivity, and every element is conserved to
1e-9. Only the primary solid forms; a second solid (a eutectic, or Lippard et al.'s carbides)
is beyond it, as are the AerMet100 data of Lippard et al. (1998), which need commercial
databases.

