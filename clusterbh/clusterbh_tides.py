from __future__ import division
import numpy
import re
from pylab import log, sqrt, pi, log10, exp, sin, tanh
from scipy.special import erf, hyp2f1
from scipy.integrate import solve_ivp, quad
numpy.seterr(invalid='ignore', divide='ignore')


"""
- Current parameters are working only for the default options. If another model is selected, a different set of values may be used.
- Default option requires the use of Simple Stellar Population. To install the package, type the command 'pip install astro-ssptools==2.0.0'. Version 2.0.0 uses Zsolar=0.02.
  More details are available at SMU-clusters/ssptools 
"""

class clusterBH:
    def __init__(self, N, rhoh, **kwargs):
        """
        Initialize the star cluster model.

        Parameters:
        - N (int): Initial number of stars.
        - rhoh (float): Half-mass density [Msun/pc^3].
        - kwargs: Additional parameters to override defaults.
        """
        
        # Physical constants.
        self.G  = 0.004499 # [pc^3 /Msun /Myr^2] Gravitational constant. It is 0.004302 in [(km/s)**2 pc/Msun].
        self.Zsolar = 0.02 # Solar metallicity.
        
        # Cluster ICs.
        self.N = N # Initial number of stars.
        self.m0 = 0.606 # Taken from CMC models. The user must specify it beforehand for a given IMF, or use the function _initial_average_mass to compute it directly.
        self.fc = 1 # Factor for escape velocity. Different for other King models. Default to W0=5, normalized. Scales roughly as sqrt(W0/5).
        self.rg = 8 # [kpc] Galactocentric distance of the cluster. For eccentric orbits, think of it as a(1 - e) where a is the semi-major axis, e is eccentricity.
        self.Z = 0.0002 # Metalicity of the cluster. Default to poor metal cluster. 
        self.omega0 = 0 # [1 / Myrs] Angular frequency of cluster due to rotation, not orbit.
        
        # BHMF. It is required only if the user does not implement the available Simpe Stellar Population (SSP) tools.
        # The current treatment for the BHMF is a simple power-law. At each iteration after core collapse, the BHMF is updated. The upper mass decreases with time, as dictated by Mbh.
        self.f0 = 0.06 # Initial fraction of BHs. It decreases with metallicity and it is important only when the SSP are not used.
        self.mlo = 3 # [Msun] Lowest BH mass in the BHMF.
        self.mup = 40 # [Msun] Upper BH mass in the BHMF
        self.alpha_BH = -0.5 # Slope in the BHMF. If we consider kicks, the slope cannot be equal to -5 -3k with k being positive integer. The special value of -2 (k=-1) is considered, but not the rest. They can be used but without kicks.
        self.fretm = 1 # Retention fraction of BHs. The default value neglects kicks.
        self.t_bhcreation = 8 # [Myrs] Time required to create all BHs.
        
        # Model parameters. 
        self.mns = 1.4 # [Msun] Mass of Neutron Stars (NS).
        self.kick_slope = 1 # Slope for kicks if the sigmoid option is selected.
        self.kick_scale = 20 # Scale for kicks.
        self.sigmans = 265 # [km/s] Velocity dispersion for NS. 
        self.gamma = 0.02 # Parameter for Coulomb logarithm.
        self.S_crit = 0.16 # Critical value for Spitzer's parameter, below which the system reached complete equipartition.
        self.Sf = 0.16 # Auxiliary value for Spitzers parameter for when equipartition is reached. Used to keep the BH ejection rate finite and nonzero. Should be equal to or below the critical value for complete equipartition.
        self.x = 3./4 # Exponent used for finite escape time from Lagrange points.
        self.r0 = 0.8 # Ratio of the half-mass radius over the Virial radius initially.
        self.rf = 1 # Final value for ratio rh / rv. Could use 0.96
        self.f = 1 # Prefactor that relates relaxation of stars with the half-mass relaxation.
        self.tcross_index = 1 # Prefactor for half-mass crossing time defined as tcr = index/sqrt(G ρ).
        self.c = 10 # Exponent for exponential tides. Can be used as additional parameter for other tidal models.
        self.c_bh = 10 # Exponent for tides used in the BH evaporation rate. 
        self.rp = 0.8 # [kpc] Constant parameter for galactic potentials.
        self.rpc = 1 # [pc] Constant parameter for cluster potentials, for instance Plummer.
        
        # Includes also parameters that are trials for additional elements / future implementations. A few are subjects to future fittings.
        #_____________________________________________
        self.a = 0 # Exponent for metallicity dependent stellar winds. Currently deactivated, impies the same mass-loss rate for different metallicities.
        self.gamma1 = 3 / 2 # Exponent for evaporation rate in the rotational part. Default option leaves xi / trh invariant, changes appear only through different evolution of rh.
        self.gamma2 = 1 # Exponent that can be used for the parametrization of the BH ejection rate.
        self.gamma3 = 3 # Constant that can be used in the balance function.
        self.gamma4 = 2 # Second constant for the balance function. Combined with gamma3, it allows for a smooth connection between the two phases.
        self.c1 = 0 # First prefactor of time dependent stellar evolution.
        self.c2 = 0 # Second prefactor of time dependent stellar evolution. Currently only a logarithmic model is considered as Lamers et al (2010).
        self.d1 = 0.18 # First exponent to describe the evolution of parameter fc.
        self.d2 = 0.14 # Second exponent to describe the evolution of parameter fc. Can be used in order to specify how W0 evolves for a given cluster as function fo M and rh.
        self.chi = 0. # Prefactor used for changing the average stellar mass wrt tides. It is important for small galactocentric distances.
        self.mst_inf = 1.4 # [Msun] Maximum upper mass for stars at infinity. It represents the mass of the remnants, taken to be close to the limit of heaviest white dwarf / lighter neutron star.
        #_________________________________________________
        
        # Parameters that were fit to N-body / Monte Carlo models. All of them are dimensionless.
        self.zeta = 0.085 # Energy loss per half-mass relaxation. It is assumed to be constant.
        self.beta = 0.0638 # Ejection rate of BHs from the core per relaxation. Taken to be universally constant for clusters indendent of initial conditions, but with a substantial BH population in the core.
        self.nu = 0.072 # Mass loss rate of stars due to stellar evolution. Here it is independent of metallicity, the same for all clusters.
        self.tsev = 1 # [Myrs]. Time instance when stars start evolving. Typically it should be a few Myrs.
        self.a0 = 1 # Fix zeroth order in ψ. Serves as the contribution of the stellar population to the half-mass mass spectrum.
        self.a11 = 1.94 # First prefactor in ψ. Relates the BH mass fraction within the half-mass to the total fraction.
        self.a12 = 0.634 # Second prefactor in ψ. Relates the mass ratio of BHs over the total average mass within the half-mass radius and the total. It is the product of a11 and a12 that matters.
        self.a3 = 0.984 # Prefactor of average mass within the half-mass radius compared to the total. The values of a11, a12 and a3 are taken for clusters with large initial relaxations.
        self.n = 1.6222 # Exponent in the power-law for tides. Depending on its value compared to 1.5, it suggests which Virial radii lose mass faster.
        self.Rht = 0.113 # Ratio of rh/rt to give correct Mdot due to tides. It is not necessarily the final value of rh/rt. Serves as a reference scale.
        self.ntrh = 1.0967 # Number of initial relaxations in order to compute core collapse instance. Marks also the transition from the unbalanced to the balanced phase.
        self.alpha_ci = 0.0 # Initial ejection rate of stars, when we have BHs. 
        self.alpha_cf = 0.0 # Final ejection rate of stars, when the BHs have been ejected (or the final binary remains) and the stellar density increases in the core. 
        self.kin = 1 # Kinetic term of evaporating stars. Used to compute the energy carried out by evaporated stars. For CMC, kin = 2 * (log(self.gamma * N) / N) ** (1 - self.x), here it is a constant.
        self.b = 2.14 # Exponent for parameter ψ. The choices are between 2 and 2.5, the former indicates the same velocity dispersion between components while the latter complete equipartition.
        self.b_min = 2 # Minimum exponent for parameter ψ.
        self.b_max = 2.26 # Maximum exponent for parameter ψ. Taken from Wang (2020), can be used for the case of a running exponent b.
        self.b0 = 2 # Exponent for parameter ψ. It relates the fraction mst / m within rh and the total fraction.
        self.b1 = 1 # Correction to exponent of fbh in parameter ψ. It appears because the properties within the half-mass radius differ.
        self.b2 = 1.04 # Exponent of fraction mbh / m in ψ. Connects properties within rh with the global. 
        self.b3 = 0.96 # Exponent of average mass within rh compared to the global average mass. All exponents are considered for the case of clusters with large relaxation.
        self.b4 = 0.17 # Exponent for the BH ejection rate. Participates after a critical value, here denoted as fbh_crit.
        self.b5 = 0.4 # Second exponent for the BH ejection rate. Participates after a critical value for the BH fraction, here denoted as qbh_crit. Its effect is deactivated for now.
        self.Mval0 = 3.1187 # Initial value for mass segregation. For a homologous distribution of stars, set it equal to 3.
        self.Mval_cc = 3.1187 # Contribution of stellar evolution in half-mass radius after core collapse. It is considered along with Henon's constant. If set equal to 2, it does not contribute. Can be neglected in future extensions if a smoothing transition prior-post core collapse is inserted.
        self.Mvalf = 3.1187 # Final parameter for mass segregation. Describes the maximum value for the level of segregation, it is universal for all clusters and appears at the moment of core collapse.
        self.p = 0.1 # Parameter for finite time stellar evaporation from the Lagrange points. Relates escape time with relaxation and crossing time. Suggests that more energetic particles leave easier the cluster.
        self.fbh_crit = 0.005 # Critical value of the BH fraction to use in the ejection rate of BHs. Decreases the fractions E / M φ0 properly, which for BH fractions close O(1)% approximately can be treated as constant.
        self.qbh_crit = 25 # [Msun] Ratio of mbh / m when the BH ejection rate starts decreasing.
        self.S0 = 1.7247 # Parameter used for describing BH ejections when we are close to equipartition. Useful for describing clusters with different metallicities using the same set of parameters, as well as for clusters with small BH populations which inevitably slow down the ejections to some extend.
        self.gamma_exp = 7.75 # Parameter used for obtaining the correct exponent for parameter psi as a function of the BH fraction. Uses an exponential fit to connect minimum and maximum values of b.
        
        # Some integration parameters.
        self.tend = 13.8e3 # [Myrs] Final time instance where we integrate to. Here taken to be a Hubble time.
        self.dtout = 2 # [Myrs] Time step for integration. Default to a few Myrs.
        self.Mst_min = 100 # [Msun] Stop criterion for stars. Below this value, the integration stops.
        self.Mbh_min = 500 # [Msun] Stop criterion for BHs.  below this value the integration stops. It is considered along with the stellar option, so only if both consitions are met, does the integrator stop.
        self.integration_method = "RK45" # Integration method. Default to a Runge-Kutta.
        
        # Output.
        self.output = False # A Boolean parameter in order to save the results of integration along with a few important quantities.
        self.outfile = "cluster.txt" # File to save the results if needed.

        # Conditions.
        self.ssp = True # Condition to use the SSP tools in order to extract the initial BH population for a given IMF.
        self.kick = True # Condition to include natal kicks. Affects the BH population obtained.   
        self.tidal = True # Condition to activate tides.
        self.escapers = False # Condition for escapers to carry negative energy as they evaporate from the cluster due to the tidal field.
        self.varying_kin = False # Condition to include a variation of the kinetic term with the number of stars. It is combined with 'escapers'.
        self.two_relaxations = True # Condition in order to have two relaxation time scales for the two components. Differentiates between ejections and evaporation.
        self.mass_segregation = False # Condition for mass segregation. If activated, parameter Mval evolves from Mval0 up until Mvalf within one relaxation.
        self.Virial_evolution = False # Condition to consider the different evolution for the Virial radius from the half-mass radius.
        self.psi_exponent_run = False # Condition to have a running exponent in parameter ψ based on the BH fraction.
        self.finite_escape_time = False # Condition in order to consider escape from Lagrange points for stars which evaporate. Introduces an additional dependence on the number of particles on the tidal field, prolongs the lifetime of clusters.
        self.running_bh_ejection_rate_1 = True # Condition for decreasing the ejection rate of BHs due to varying E / M φ0. Default option considers the effect of decreasing fbh only.
        self.running_bh_ejection_rate_2 = True # Condition for decreasing the ejection rate of BHs due to equipartition.
        self.running_stellar_ejection_rate = False # Condition for changing the stellar ejection rate compared to the tidal field.
        self.rotation = False # Condition to describe rotating clusters. Currently it is inserted as a simple extension, more study is required.
        self.dark_clusters = False # Condition for describing dark clusters. When used, the approximations in ψ are not used. it is considered as future extension and needs to be augmented.
        self.sev_Z_run = False # Condition to have a running parameter ν with respect to metallicity. A power-law option is available however no significant dependence on Z is observed in the models studied so far.
        
        # Stellar evolution model.
        self.sev_model = 'constant' # Default option is constant mass loss rate.
        
        # Motion of cluster.
        self.Vc = 220. # [km/s] Circular velocity of cluster, for instance inside a singular isothermal sphere (SIS). 
        
        # Galactic model.
        self.galactic_model, self.rt_index = 'SIS', 2  # Default to 'SIS'. Affects the tidal radius.
        
        # Tidal model.
        self.tidal_model = 'Power_Law'  # Default to 'Power_Law'. Affects tidal mass loss and thus the evolution of the half-mass radius.
        
        # Cluster model.
        self.cluster_model = 'Point_mass' # Default to 'Point_mass'. Currently it affects only the energy contribution of evaporating stars, if activated.
        
        # Model for the BH ejection rate beta.
        self.beta_model = 'exponential' # Default option to the exponential model. Affects the dependence of the BH ejection rate on Spitzer's parameter and thus the extrapolation to small BH populations or large metallicity clusters.
        
        # Balance model. It can be used for a smooth connection of the unbalanced phace with post core collapse.
        self.balance_model = 'error_function' # Default option to error function. Uses a function in order to smoothly connect the unbalanced phase to post core collapse. Serves as a future implementation to make the differential equations continuous.
        
        # IMF. For top heavy, fine-tune a3. Possible extensions to a_slope4, m_break5, nbin4 and more are available as well.
        self.a_slope1 = -1.3 # Slope of mass function for the first interval.
        self.a_slope2 = -2.3 # Slope of mass function for the second interval.
        self.a_slope3 = -2.3 # Slope of mass function for the third interval. All intervals are dictated by the mass breaks.
        self.m_break1 = 0.08 # [Msun] Lowest stellar mass of the cluster.
        self.m_break2 = 0.5 # [Msun] Highest stellar mass in the first interval.
        self.m_break3 = 1. # [Msun] Highest stellar mass in the second interval.
        self.m_break4 = 150. # [Msun] Highest mass in the cluster. Default option for 4 mass breaks, otherwise the user can specify which is the largest mass.
        self.nbin1 = 5 # Number of bins in the first interval.
        self.nbin2 = 5 # Number of bins in the second interval.
        self.nbin3 = 20 # Number of bins in the third interval.
        # Mass breaks define the intervals, slopes the exponent in the IMF in each interval, indicating the probability of forming such star. Bins are used to improve the numerical resolution of the IMF and ensure accurate sampling.
        
        # Default options for extracting the BH population.
        self.BH_IFMR = 'banerjee20' # Default option for the BH IFMR.
        self.WD_IFMR = 'mist18' # Default option for the WD IFMR
        self.kick_method = 'maxwellian' # Default option for kicks, if activated.
        self.binning_method = 'default' # Default option for binning.
        
        # Check input parameters. Afterwards we start computations.
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        # Define models for stellar evolution.
        self.sev_dict={
            'constant': lambda t: self.nu,
            'exponential': lambda t: self.nu * exp (- self.c1 * ((t - self.tsev) / self.tsev) ** self.c2),
            'power_law': lambda t: self.nu / (1 + self.c1 * ((t - self.tsev) / self.tsev) ** self.c2),
            'logarithmic': lambda t: self.nu * (1 + self.c1 * log(t) - self.c2 * log(t) ** 2)
        }
        
        
        # Galactic properties.
        self.Mg = 1.023 ** 2 * self.Vc ** 2 * self.rg * 1e3 / self.G # [Msun] Mass of the galaxy inside the orbit of the cluster. It is derived from orbital properties.
        self.L = 1.023 * self.Vc * self.rg * 1e3 # [pc^2/Myrs] Angular momentum.
        
        # Define the tidal models dictionary.
        self.tidal_models = { # Both the half-mass and the tidal radius should be in [pc]. All the rest parameters are fixed. In a different scenario, they should be treated as variables.
            'Power_Law': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n, 
            'Exponential': lambda rh, rt: 3 * self.zeta / 5 * exp(self.c * rh / rt),
            'Logarithmic': lambda rh, rt: 3 * self.zeta / 5 * log(1 + self.c * rh / rt),
            'Hybrid1': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n * exp(- self.c * rh / rt),
            'Hybrid2': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n * log(1 + self.c * rh / rt),
            'Saturated': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n * (1 - (rh / rt / self.Rht) ** self.c)
        }
        
        # Galactic model dictionary.
        self.galactic_model_dict = {  # We create a dictionary for spherically symmetric galactic potentials. A model can then be selected from the dictionary. The index is used for the tidal radius only and it is dimensionless. The potential has units are [pc^2 / Myr^2]. Radius should be in [kpc], mass in [Msun].
            'SIS': {'rt_index':2 , 'phi':lambda r, Mg: self.Vc ** 2 * log(r / self.rp)}, # Singular isothermal sphere. It is the default.
            'Point_mass': {'rt_index':3, 'phi': lambda r, Mg: - 1e-3 * self.G * Mg / r }, # Point mass galaxy.
            'Hernquist': {'rt_index':(3 * self.rg + self.rp) / (self.rg + self.rp), 'phi': lambda r, Mg: - 1e-3 * self.G * Mg / (r + self.rp) },  # Hernquist model.
            'Plummer': {'rt_index':3 * self.rg ** 2 / (self.rg ** 2 + self.rp ** 2), 'phi':lambda r, Mg: - 1e-3 * self.G * Mg / sqrt(r ** 2 + self.rp ** 2)},  # Plummer model.
            'Jaffe': {'rt_index':(3 * self.rg + 2 * self.rp) / (self.rg + self.rp), 'phi':lambda r, Mg: - 1e-3 * self.G * Mg / self.rp * log(1 + self.rp / r)},   # Jaffe model.
            'NFW': {'rt_index':1 + (2 * log(1 + self.rg / self.rp) - 2 * self.rg / (self.rg + self.rp) - (self.rg / (self.rg + self.rp)) ** 2 ) / (log(1 + self.rg / self.rp) - self.rg / (self.rg + self.rp)), 'phi':lambda r, Mg:- 1e-3 * self.G * Mg / r * log(1 + r / self.rp)}, # Navarro-Frank-White model.
            'Isochrone': {'rt_index':1 + 1 / (self.rp + sqrt(self.rg ** 2 + self.rp ** 2)) / sqrt(self.rg ** 2 + self.rp ** 2) * (2 * self.rg ** 2 - self.rp ** 2 * (self.rp + sqrt(self.rg ** 2 + self.rp ** 2)) / sqrt(self.rp ** 2 + self.rg ** 2)), 'phi':lambda r, Mg:- 1e-3 * self.G * Mg / (self.rp + sqrt(r ** 2 + self.rp ** 2))}
        } # Exclude the Miyamoto-Nagai potential for now since it is not spherically symmetric.
        
        # Cluster model dictionary.
        self.cluster_model_dict = { # We create a dictionary for spherically symmetric cluster potentials. Units are [pc^2 / Myr^2]. Radius should be in [pc], mass in [Msun].
            'SIS': lambda r, M: self.G * M / r * log(r / self.rpc),
            'Point_mass': lambda r, M: - self.G * M / r,
            'Plummer': lambda r, M: - self.G * M / sqrt(r ** 2 + self.rpc ** 2),
            'Hernquist': lambda r, M: - self.G * M / (r + self.rpc),
            'Jaffe': lambda r, M: - self.G * M / self.rpc * log(1 + self.rpc / r),
            'NFW': lambda r, M: - self.G * M / r * log(1 + r / self.rpc),
            'Isochrone': lambda r, M: - self.G * M / (self.rpc + sqrt(r ** 2 + self.rpc ** 2))
        }
        
        # Introduce dictionaries for essential properties to derive the BH population for a given set of initial conditions. Contain strings similar to their names. They serve as input to get the BH population.
        
        # Dictionary for the BH IFMR. 
        self.BH_IFMR_dict = {
            'banerjee20' : 'banerjee20', 'cosmic-rapid': 'cosmic-rapid',
            'cosmic-delayed': 'cosmic-delayed', 'linear': 'linear', 
            'powerlaw': 'powerlaw','brokenpowerlaw': 'brokenpowerlaw'}
        
        # Dictionary for the WD IFMR.
        self.WD_IFMR_dict = {'mist18':'mist18', 'linear':'linear'}
        
        # Dictionary for kicks.
        self.kick_dict = {'maxwellian' : 'maxwellian', 'sigmoid' : 'sigmoid'}
        
        # Dictionary for binning.
        self.binning_dict = {'default' : 'default', 'split_log': 'split_log', 'split_linear': 'split_linear'}
        
        # Dictionary that introduces a scaling on the BH ejection rate with respect to Spitzer's paramater. It can be used as a proxy for a different parametrizations as well.
        self.beta_dict = {
           'exponential': lambda S: 1 - exp(- (S / self.S0) ** self.gamma2),
           'logistic1': lambda S: (S / (S + self.S0)) ** self.gamma2,
           'logistic2': lambda S: (S ** self.gamma2 / (S ** self.gamma2 + self.S0 ** self.gamma2)),
           'error_function': lambda S:  erf((S + self.S_crit) / self.S0),
           'hyperbolic': lambda S: tanh((S + self.S_crit) / self.S0),
           'trigonometric': lambda S: sin(pi / 2 * S / (S + self.S0))
        }
        
        # Dictionary for balancing functions. It is currently on trial stage and is not used. Functions must be 0 at t=0 and for initial times, and 1 at tcc<=t.
        self.balance_dict = {
            'error_function':lambda t:  0.5 * (1 + erf(self.gamma4 * self.gamma3 * (t - self.gamma4 * self.tcc / self.gamma3 ) / self.tcc)) ,
            'hyperbolic': lambda t: 0.5 * (1 + tanh(self.gamma3 * self.gamma4 * (t - self.gamma4 * self.tcc / self.gamma3) / self.tcc)),
            'exponential': lambda t: 1 - exp(- self.gamma3 * (t / self.tcc) ** self.gamma4)
        }
        
        # Check the available IMF. The default option is 3 slopes, bins and break masses.
        self.a_slopes, self.m_breaks, self.nbins = self._IMF_read() # The user is allowed to include other regions in the IMF, for instance m > 150 with a steeper slope, -2.7.
        
     #   self.m0 = self._initial_average_mass(self.a_slopes, self.m_breaks) # Average mass obtained for this particular IMF. 
        
        self.FeH = log10(self.Z / self.Zsolar) # Metallicity in solar units.
        
        self.M0 = self.m0 * N # [Msun] Total mass of stars (cluster) initially. This is used for computations, even if the BH population is present in clusterBH at t=0.
        self.rh0 = (3 * self.M0 / (8 * pi * rhoh)) ** (1./3) # [pc] Half-mass radius initially.
        
        self.vesc0 = self._vesc(self.M0, self.rh0) # [km/s] Initial escape velocity.
        
        self.mb = (9 * pi / 2) ** (1./6) * self.sigmans * self.mns / self.vesc0 # [Msun] Lowest mass unaffected by kicks. All masses below this threshold are effected by kicks.
        
        # Check the IFMR model used in the SSP tools. If anything is wrong, raise error.
        if (self.ssp):
            if hasattr(self, 'BH_IFMR') and self.BH_IFMR in self.BH_IFMR_dict:
                self.BH_IFMR_method = self.BH_IFMR_dict[self.BH_IFMR]
            else:
                raise ValueError(f"Invalid BH IFMR: {self.BH_IFMR}.")
            
            if hasattr(self, 'WD_IFMR') and self.WD_IFMR in self.WD_IFMR_dict:
                self.WD_IFMR_method = self.WD_IFMR_dict[self.WD_IFMR]
            else:
                raise ValueError(f"Invalid BH IFMR: {self.BH_IFMR}.")
        
            if hasattr(self, 'kick_method') and self.kick_method in self.kick_dict:
                self.kick_method = self.kick_dict[self.kick_method]
            else:
                raise ValueError(f"Invalid kick method: {self.kick_method}.")
        
            if hasattr(self, 'binning_method')  and self.binning_method in self.binning_dict:
                self.binning_method = self.binning_dict[self.binning_method]
            else:
                raise ValueError(f"Invalid binning method: {self.binning_method}.")
        
        # Check the BH ejection model.
        if not hasattr(self, 'sev_model') or self.sev_model not in self.sev_dict:
            raise ValueError(f"Invalid model for stellar evolution: {self.sev_model}.")
        
        # Check the BH ejection model.
        if not hasattr(self, 'beta_model') or self.beta_model not in self.beta_dict:
            raise ValueError(f"Invalid model for BH ejections: {self.beta_model}.")
        
        # Check the balance model. Future implementation.
        if not hasattr(self, 'balance_model') or self.balance_model not in self.balance_dict:
            raise ValueError(f"Invalid model for balancing: {self.balance_model}.")
        
        # Check if we use SSP for the BHMF.
        if (self.ssp):
            import ssptools # Import the package. Must be installed first.
            # Implement kicks, if activated, for this IMF, number of stars, with such metallicity and central escape velocity.
            self.ibh = ssptools.InitialBHPopulation.from_powerlaw(self.m_breaks, self.a_slopes, self.nbins, self.FeH, N0=N, vesc=self.vesc0, natal_kicks=self.kick, BH_IFMR_method=self.BH_IFMR_method, WD_IFMR_method=self.WD_IFMR_method,
                                                              kick_method=self.kick_method, binning_method=self.binning_method, kick_slope=self.kick_slope, kick_scale=self.kick_scale, kick_vdisp=self.sigmans) # Version 2 of SSP tools. Uses solar metallicity Zsolar = 0.02.
        
            self.Mbh0 = self.ibh.Mtot  # [Msun] Expected initial mass of BHs due to kicks.
            self.f0 = self.Mbh0 / self.M0 # Initial fraction of BHs. It should be close to 0.05 for poor-metal clusters. 
            
            self.Nbh0 = numpy.round(self.ibh.Ntot).astype(int) # Initial number of BHs. We round the number for consistency.
            self.mbh0 = self.Mbh0 / self.Nbh0 # [Msun] Initial average BH mass.
            self.Mst_lost = self.ibh.Ms_lost # [Msun] Mass of stars lost in order to form BHs.
            self.t_bhcreation = self.ibh.age # [Myrs] Time needed to form these BHs.
            
        # Should the user wishes to exclude the ssp tools and use another approach, they can first define f0 and then compute Mbh0. A simple approach would be the same value regardless of kicks.
        else:
            if (self.kick):
               
               qul, qub, qlb = self.mup / self.mlo, self.mup / self.mb, self.mlo / self.mb # Mass rations.

               if self.alpha_BH != -2: # It is important so that the hypergeometric function is well defined.
                   b = (self.alpha_BH + 2) / 3 # Auxiliary exponent.
                   h1 = hyp2f1(1, b, b + 1, -qub ** 3) # Hypergeometric function, part of the solution.
                   h2 = hyp2f1(1, b, b + 1, -qlb ** 3) # Hypergeometric function, part of the solution.
                   self.fretm = 1 - (qul ** (3 * b) * h1 - h2) / (qul ** (3 * b) - 1) # Retention fraction.
               
               else:
                   
                   self.fretm = log( (qub ** 3 + 1) / (qlb ** 3 + 1) ) / log(qul ** 3) # Retention fraction.
                  
            self.Mbh0 = self.fretm * self.f0 * self.M0 # [Msun] Prediction for the initial BH mass if SSP are not used.
            self.mbh0 = self._mbh(self.Mbh0) # [Msun] Initial average BH mass.
            self.Nbh0 = numpy.round(self.Mbh0 / self.mbh0).astype(int) # Initial number of BHs.
            self.Mst_lost = (self.m0 - self._mst_instance(self.t_bhcreation)) * N # [Msun] Approximate value for the stellar mass that was lost in order to create all BHs.
        
        # Initial relaxation is extracted with respect to the BH population that will be obtained. It is an approximation such that large metallicity clusters have larger relaxation, as expected. This approach inserts an indirect metallicity dependence.
        self.psi0 = self._psi(self.Mbh0 / (self.M0 + self.Mbh0 - self.Mst_lost), self.M0 + self.Mbh0 - self.Mst_lost, self.mbh0, self._mst_instance(self.t_bhcreation)) # Use as time instance the moment when all BHs are formed. In this regime, tides are not so important so all the mass-loss comes from stellar evolution. 
        self.trh0 = 0.138 / (self.m0 * self.psi0 * log(self.gamma * self.N)) * sqrt(self.M0 * self.rh0 ** 3 / self.G ) * (1 - 2 * self.omega0 ** 2 * self.rh0 ** 3 / (self.G * self.M0)) ** (3 / 2) # [Myrs] We use m0 since at t=0 the mass distribution is the same everywhere.
        self.tcc = self.ntrh * self.trh0 # [Myrs] Core collapse. Effectively it depends on metallicity, size and Mass of the cluster.
        
        # Check if the galactic model, cluster model, tidal model have changed. Based on the available choices in the dictionaries, change the values accordingly.
        if (self.tidal): # Valid check only if we have tides.
            if hasattr(self, 'galactic_model') and self.galactic_model in self.galactic_model_dict:
                self.rt_index = self.galactic_model_dict[self.galactic_model]['rt_index']
            else:
                raise ValueError(f"Invalid galactic model: {self.galactic_model}.") 
        
            if not hasattr(self, 'cluster_model') or self.cluster_model not in self.cluster_model_dict:
                raise ValueError(f"Invalid cluster model: {self.cluster_model}.") 
        
            if not hasattr(self, 'tidal_model') or self.tidal_model not in self.tidal_models:
                raise ValueError(f"Invalid tidal model: {self.tidal_model}.")
         
        self.evolve(N, rhoh) # Runs the integrator, generates the results.
    
    def _IMF_read(self):
        """
        Processes and validates class attributes related to the initial mass function (IMF).

        - Extracts `a_slopeX`, `m_breakX` [Msun], and `nbinX` attributes dynamically. The default number are 3, 4, 3 but the user is allowed to insert more. They should be in a sequence.
        - Sorts these attributes by their numeric suffix.
        - Validates `m_breakX` to ensure values are strictly increasing, stopping at the first invalid entry, if any.
        - Adjusts `a_slopeX` and `nbinX` lists to align with the number of valid `m_breakX` values.

        Returns:
            tuple:
            - a_slopes (list): Adjusted slope values.
            - valid_m_breaks (list): Validated and sorted mass break values.
            - nbins (list): Adjusted bin values.
        """
        
        # Extract and sort attributes dynamically. The user is allowed to insert more slopes, masses and bins. They must however be in order.
        a_slopes = [(attr, getattr(self, attr)) for attr in dir(self) if re.match(r"a_slope\d+$", attr)]
        m_breaks = [(attr, getattr(self, attr)) for attr in dir(self) if re.match(r"m_break\d+$", attr)]
        nbins = [(attr, getattr(self, attr)) for attr in dir(self) if re.match(r"nbin\d+$", attr)]

        # Sort by the numeric suffix.
        a_slopes = [value for _, value in sorted(a_slopes, key=lambda x: int(re.search(r"\d+$", x[0]).group()))]
        m_breaks = [value for _, value in sorted(m_breaks, key=lambda x: int(re.search(r"\d+$", x[0]).group()))]
        nbins = [value for _, value in sorted(nbins, key=lambda x: int(re.search(r"\d+$", x[0]).group()))]

        # Validate m_breaks for increasing order. If one mass is smaller than the previous one, we stop. It is a safety measure.
        valid_m_breaks = []
        for i, m_break in enumerate(m_breaks):
            if i == 0 or m_break > valid_m_breaks[-1]:
                valid_m_breaks.append(m_break)
            else:
                break

        # Adjust a_slopes and nbins lengths. They should be smaller than the available masses.
        self.max_m_break = valid_m_breaks[-1] # [Msun] Maximum mass in the IMF.
        num_valid_m_breaks = len(valid_m_breaks) 
        num_slopes_bins = num_valid_m_breaks - 1 
        
        # Ensure we have the right number of slopes and bins for each mass range
        if len(a_slopes) != num_slopes_bins:
           raise ValueError(f"Number of slopes must be {num_slopes_bins} for {num_valid_m_breaks} mass break points. Please provide the correct number of slopes.")
    
        if len(nbins) != num_slopes_bins:
           raise ValueError(f"Number of bins must be {num_slopes_bins} for {num_valid_m_breaks} mass break points. Please provide the correct number of bins.")
    

        a_slopes = a_slopes[:num_slopes_bins]
        nbins = nbins[:num_slopes_bins]

        # Return the validated arrays
        return a_slopes, valid_m_breaks, nbins

    # Compute initial average mass.
    def _initial_average_mass(self, a_slopes, m_breaks):
       """
       Calculate the initial average mass of stars based on a piecewise power-law IMF.
    
       Parameters:
       -----------
       a_slopes : list of floats
          Slopes of the IMF in different mass ranges. Each slope corresponds to a segment of the piecewise IMF.
       m_breaks : list of floats
          Breakpoints of the mass ranges [Msun]. The length of this list should be one more than `a_slopes`.

       Returns:
       --------
       float
        The average mass of stars calculated using the IMF and its normalization [Msun].
    
       Notes:
       ------
       - This function integrates the IMF.
       - The function assumes the IMF is continuous across the mass breakpoints.
       - Special handling is included for slopes to avoid division by zero.
       """
       def integrate_IMF(m_low, m_high, p):
        
           if p == -1:  # Special case where p = -1 to avoid division by zero, should the user select such value.
              return log(m_high / m_low) 
           else:
              return (m_high ** (p + 1) - m_low ** (p + 1)) / (p + 1)
       
       # Normalization constant to make the IMF continuous.
       c_values = [1.0]  # c1 = 1.0
       normalization_constants = []
       
       # Calculate the rest of the c values based on slopes and mass breakpoints. This is for continuity.
       for i in range(1, len(a_slopes)):
           c_values.append(m_breaks[i] ** (a_slopes[i - 1] - a_slopes[i]))

       # Compute the cumulative products to form the normalization_constants array.
       for i in range(len(c_values) - 1):
         normalization_constants.append(c_values[i] * c_values[i + 1])
         
       normalization_constants.insert(0, c_values[0]) # First entry is 1. It does not matter because we essentially compute the normalised prefactors.
       
       stellar_mass = 0
       stellar_number = 0
       
       for i, a in enumerate(a_slopes):
          m_low = m_breaks[i]
          m_high = m_breaks[i + 1]
          c_i = normalization_constants[i]
          
          # Compute numerator 
          stellar_mass += c_i * integrate_IMF(m_low, m_high, a + 1)
         
          # Compute denominator
          stellar_number += c_i * integrate_IMF(m_low, m_high, a)
    
       # Calculate the average mass
       return stellar_mass / stellar_number
    
    
    # Function to determine by how much the BHMF changes, given a particular mass loss.
    def _deplete_BHMF(self, M_eject, M_BH, N_BH):
        """
        Deplete the BHMF by ejecting mass starting from the heaviest bins.
    
        Parameters:
         -----------
        M_eject : float
           Total mass to eject from the BHMF [Msun].
        M_BH : numpy.ndarray
           Array representing the total mass in each mass bin [Msun].
        N_BH : numpy.ndarray
           Array representing the number of BHs in each mass bin.

        Returns:
        --------
        Tuple of numpy.ndarray
           Updated arrays for the mass in each BH bin (M_BH) and the number of BHs (N_BH).
    
        Notes:
        ------
        - The function starts ejecting mass from the heaviest bin and proceeds to lighter bins.
        - If the mass to eject (`M_eject`) exceeds the total mass in a bin, the entire bin is depleted.
        - For partially depleted bins, the corresponding number of BHs (`N_BH`) is adjusted proportionally.
        - The method avoids modifying the original arrays by creating copies of `M_BH` and `N_BH`.
        """
        # Avoid altering initial BH bins.
        M_BH, N_BH = M_BH.copy(), N_BH.copy() # [Msun, dimensionless]

        # Remove BH starting from Heavy to Light.
        j = M_BH.size

        while M_eject != 0: # [Msun]
            j -= 1

            # Stop ejecting if trying to eject more mass than there is in BHs.
            if j < 0:
                break
             
            # Remove entirety of this mass bin.
            if M_BH[j] < M_eject: # [Msun]
                M_eject -= M_BH[j] # [Msun]
                M_BH[j] = 0 # [Msun]
                N_BH[j] = 0 # Dimensionless
                continue

            # Remove required fraction of the last affected bin.
            else:
                m_BH_j = M_BH[j] / N_BH[j] # [Msun]
                M_BH[j] -= M_eject # [Msun]
                N_BH[j] -= M_eject / m_BH_j # Dimensionless

                break

        return M_BH, N_BH
    
    def _find_mmax(self, Mbh):
        """
        Computes the maximum BH mass at each time instance for two cases of BHMFs. Either directly from the SSP tools, or from a simple power-law the user has specified.
    
        Parameters:
        -----------
        Mbh : float
            The current total BH mass [Msun].
    
        Returns:
        --------
        mmax : float
            The maximum individual BH mass at the current time instance [Msun]. It changes only by assuming ejections. A different prescription is needed if the tidal field is important for the BH subsystem.
        """
        
        if (self.ssp):
            M_eject = self.Mbh0 - Mbh # [Msun] Mass that has been ejected.
            M_BH, N_BH = self._deplete_BHMF(M_eject, self.ibh.M, self.ibh.N) # [Msun], dimensionless. BHMF at a given time instance.
            valid_bins = N_BH > 0 # Create a mask to see which bins are not empty.
            
            # Find the maximum individual BH mass in the valid bins.
            mbh = self._mbh(Mbh) # [Msun] Average BH mass.
            
            if Mbh < mbh: # This means that we have effectively ejected all BHs.
                mmax = 0
            else:
                mmax = max(self.ibh.m[valid_bins])  # [Msun] Consider only non-zero bins.
                
            return mmax
        
        else:
            if Mbh < self.mlo: # If we have no BHs essentially, we should set this equal to 0.
                mmax = 0
                return mmax
            
            a2 = self.alpha_BH + 2 # Auxiliary exponent.
        
            if (self.kick):
                def integr(mm, qmb, qlb):
                    a2 = self.alpha_BH + 2
                    if a2 == 0: # This excluded only the alpha_BH = -2, but in principle other choices should be excluded, those that are problematic for hypergeometric functions (negative integers for b + 1). We neglect those because they make the BHMF quite steep.
                       return log((1 + qmb ** 3) / (1 + qlb ** 3))  # Integral of power-law mass function without the constant prefactor.
                    else:
                       b = a2 / 3 # Auxiliary exponent
                       h1 = hyp2f1(1, b, b + 1, -qmb ** 3) # Hypergeometric function used for the solution.
                       h2 = hyp2f1(1, b, b + 1, -qlb ** 3) # Hypergeometric function used for the solution.
                       return mm ** a2 * (1 - h1) - self.mlo ** a2 * (1 - h2) # BH mass extracted by integrating the mass function, neglecting the constant prefactor.

                # Solve with respect to the maximum BH mass.
                N_points = 500 # Create points for mmax, to see which value is closer to describing the total BH mass a given time instance.
                mmax_ = numpy.linspace(self.mlo, self.mup, N_points) # List of possible values for the maximum BH mass at a given time instance.
                qmb, qlb  = mmax_ / self.mb, self.mlo / self.mb # Auxiliary mass rations that can be used in the solution.

                A = self.Mbh0 / integr(self.mup, self.mup / self.mb, qlb) # Constant prefactor from the mass function.

                Mbh_ = A * integr(mmax_, qmb, qlb) # BH mass points that are generated from the possible max BH mass points.
                mmax = numpy.interp(Mbh, Mbh_, mmax_) # Extract the maximum BH mass in the IMF at any time instance.
            else:
                if self.alpha_BH != -2: # Check the exponent for the BHMF and for the case of no kicks, a solution can be extracted.
                   mmax = (Mbh / self.Mbh0 * (self.mup ** a2 - self.mlo ** a2) + self.mlo ** a2) ** (1./a2)
                else:
                   mmax = self.mlo * (self.mup / self.mlo) ** (Mbh / self.Mbh0) # [Msun] Maximum mass in the BHMF for the case of that exponent.
                
            return mmax
    
    # Average BH mass.
    def _mbh(self, Mbh):
        """
        Calculate the updated average BH mass after ejecting a specified amount of BH mass. 
        The computation is done based on which BHMF is used (either from the SSP or a simple power-law specified by the user).
    
        Parameters:
        -----------
        Mbh : float
          Total BH mass after ejection [Msun].

        Returns:
        --------
        float
          Average BH mass after ejections [Msun].
    
        Notes:
        ------
        - The function first calculates the total mass to be ejected (`M_eject`) in [Msun].
        - It then updates the BHMF by removing mass using `_deplete_BHMF`. The removal it top-down.
        - Finally, the average mass of the remaining BHs is computed by dividing the total mass by the total number. Units in [Msun].
        """
        # Determine amount of BH mass (total) to eject.
        M_eject = self.Mbh0 - Mbh # [Msun]
        
        if (self.ssp):
           # Deplete the initial BH MF based on this M_eject.
           M_BH, N_BH = self._deplete_BHMF(M_eject, self.ibh.M, self.ibh.N) # [Msun, dimensionless]
          
           if N_BH.sum() == 0: return 1e-99
           
           return M_BH.sum() / N_BH.sum() # [Msun]
        
        else:
            mmax = self._find_mmax(Mbh) # [Msun] Upper BH mass in the BHMF. Assumes a simple power-law function for the MF with slope a_BH.
            
            def numerator_integrand(m):
                return m ** (2 + self.alpha_BH)
        
            def denominator_integrand(m):
                return m ** (1 + self.alpha_BH)
        
            # Perform the integration using scipy's quad
            numerator, _ = quad(numerator_integrand, self.mlo, mmax, epsabs=1e-9, epsrel=1e-9)
            denominator, _ = quad(denominator_integrand, self.mlo, mmax, epsabs=1e-9, epsrel=1e-9)
            
            if denominator <= 0: return 1e-99
            
            return numerator / denominator # [Msun]
       
    # Tidal radius. This is the expression for SIS. For other gravitational potentials, other expressions may be needed.
    def _rt(self, M): 
        """
        Calculate the tidal radius of a star cluster based on its mass and orbital properties.

        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].

        Returns:
        --------
        float
           Tidal radius of the cluster [pc].

        Notes:
        ------
        - The tidal radius depends on the cluster's mass (`M`) [Msun], the angular velocity squared (`O2`) [1/Myrs^2], and a scaling parameter (`rt_index`) differing between galactic potentials.
        - The formula assumes a spherically symmetric potential, applicable to both circular and eccentric orbits (using an effective distance for the latter).
        """
        
        # Angular velocity squared.
        O2 = (self.Vc * 1.023 / (self.rg * 1e3)) ** 2 # [ 1 / Myrs^2]
        
        # Tidal radius.
        rt = (self.G * M / (self.rt_index * O2)) ** (1./3) # [pc] The expression is valid as long as the potential is spherically symmetric, regardless of the orbit.
        
        return rt
    
    
    def _mst_instance(self, t0): # Can be used to compute mst considering only stellar evolution, at a given time instance. Avoid interpolating values.
        """
        Solves the differential equation for mst over the time interval [0, t0],
        considering mst = m0 for t < self.tsev. It returns the final value only.

        Parameters:
        -----------
        t float:
            Current time (not used in this specific case).
        t0 (float):
            End time for the integration.
        
        Returns:
        -------
        float: The value of mst at t=t0.
        """
        # Check if t0 < self.tsev; if so, mst is m0. Stellar evolution hasn't started.
        if t0 < self.tsev:
            return self.m0

        # Define the differential equation for t >= self.tsev.
        def mstdot(t, mst):
            if t < self.tsev:
                return 0  # Ensure mst remains constant before self.tsev.
            return -self._nu(self.Z, t) * mst / t

        # Solve the differential equation using solve_ivp.
        sol = solve_ivp(mstdot, [self.tsev, t0], [self.m0], method=self.integration_method, t_eval=[t0], rtol=1e-8, atol=1e-10) # Accurate up until the 7th decimal.
    
        # Return the value of mst at t=t0
        return sol.y[0][-1]

    # Average mass of stars, due to stellar evolution.    
    def _mst_sev(self):
        """
        Solves the differential equation for mst over the time interval,
        considering mst = m0 for t < self.tsev. It returns the final value only.

        
        Returns:
        -------
        Array: The average stellar mass [Msun] as it would evolve only through stellar evolution.
        """
        
        # Define the differential equation for t >= self.tsev.
        def mstdot(t, mst):
            if t < self.tsev:
                return 0  # Ensure mst remains constant before self.tsev.
            return -self._nu(self.Z, t) * mst / t

        # Solve the differential equation using solve_ivp.
        t_eval = numpy.arange(0, self.tend, self.dtout) if self.dtout is not None else None # [Myrs] Time steps to which the solution will be computed.
        sol = solve_ivp(mstdot, [0, self.tend], [self.m0], method=self.integration_method, t_eval=t_eval, rtol=1e-8, atol=1e-10) # Accurate up until the 7th decimal.
    
        # Return the value of mst at each time instance as an array.
        return sol.y[0]
        
    # Friction term ψ in the relaxation. It is characteristic of mass spectrum, here due to BHs. We neglect O(2) corrections in the approximate form.
    def _psi(self, fbh, M, mbh, mst): 
        """
        Calculate psi for the cluster based on various parameters.

        Parameters:
        -----------
        fbh : float
           A factor representing the BH fraction in the range [0, 1].
    
        M : float
           Total mass of the cluster, in solar masses [Msun].
    
        mbh : float
           Average BH mass in solar masses [Msun].
    
        mst : float
           Average stellar mass [Msun].

        Returns:
        --------
        float
           The calculated psi value, which depends on the BH fraction, cluster mass and
           average masses.

        Notes:
        ------
        - If the BH mass is less than the average BH mass (mbh), 
          it indicates that BHs have been ejected, and the function returns a constant value `a0` for psi. It suggests that the
          remaining stars have roughly the same masses so no mass spectrum within rh.
        - The number of particles `Np` is calculated using the `_N` function, which depends on the cluster's mass, 
          the BH fraction, and the average masses of the populations.
        - The average stellar mass `mav` is derived by dividing the total cluster mass by the number of particles.
        - The exponent `gamma` is either a constant value `b` or it evolves based on the BH fraction (if `psi_exponent_run` is True),
          starting from an equal velocity dispersion and evolving towards equipartition as time passes.
        - The expression for `psi` includes a combination of parameters like `a0`, `a11`, `a12`, `b1`, `b2`, and others, 
          which relate the properties within the cluster's half-mass radius (`rh`) to the global properties of the cluster.
        - For dark clusters (`dark_clusters` is True), a more complex expression for `psi` is used, 
          incorporating contributions from both the stellar and BH populations.
        - From t=0 until tcc, a better expression may be needed, one that accounts for initial stellar mass spetrum and how the two populations evolve up until tcc. 
        """
        if M * fbh < mbh : return self.a0 # This means that BHs have been ejected so we have no psi.
        
        # Number of particles.
        Np = self._N(M, fbh, mbh, mst)
       
        # Average mass and stellar mass.
        mav = M / Np # [Msun]
       
        # Exponent.
        gamma = self.b # Assume that it is constant trhoughout the evolution of the cluster.
        if (self.psi_exponent_run):
           gamma = self.b_min + (self.b_max - self.b_min) * exp (- self.gamma_exp * fbh) # No BHs, starts with equal velocity dispersion. As time flows by it approaches equipartition. Maximum value is 2.26.
        
        # Approximate form. Parameters a11, a12, b1 and b2 relate the properties within rh to the global properties.
        psi = self.a0  + self.a11 * (self.a12) ** (gamma - 1) * fbh ** self.b1 * (mbh / mav) ** ((gamma - 1) * self.b2)  # This statement is needed otherwise mbh gets nan values and cannot give correct results.
        
        # Total expression. Can be used for instance if the stellar population depletes first.
        if (self.dark_clusters):
           psi = self.a0 * (mst / mav) ** (self. b0 * (gamma - 1)) * (1 - self. a11 * fbh ** self.b1) + self.a11 * self.a12 ** (gamma - 1) * fbh ** self.b1 * (mbh / mav) ** (self.b2 * (gamma - 1)) # Complete expression if we include the contribution from stars as well. We have self.a0=1.
       
        return psi
    
    # Number of particles, approximately stars for small fbh. We write it as a function of the total mass and the BH fraction. The number of BHs is a minor correction since typically 1 star in 1e3 becomes a BH, so clusters with 1e6 stars have 1e3 BHs roughly speaking, a small correction.    
    def _N(self, M, fbh, mbh, mst):
        """
        Calculate the number of particles in the cluster, including both BHs and stars.

        Parameters:
        -----------
        M : float
           Total mass of the cluster, in solar masses [Msun].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass [Msun].
    
        mst : float
           Average stellar mass [Msun].

        Returns:
        --------
        float
          The total number of particles in the cluster, considering both BHs and stars.

        Notes:
        ------
        - The number of BHs is computed by dividing the total BH mass (`M * fbh`)
          by the average BH mass (`mbh`). Both are in [Msun].
        - If the BH mass (`M * fbh`) is less than the average BH mass (`mbh`), the black hole fraction is set to zero.
        - The number of stars is calculated by dividing the remaining mass (`M * (1 - fbh)`) by the average stellar mass (`mst`).
        """
        # Number of particles.
        Np = 0
        
        # Include BHs if we have them.
        if M * fbh < mbh : fbh, mbh = 0, 1e-99
        
        Np += M * fbh / mbh 

        # Now we consider the number of stars.
        Np += M *  (1 - fbh) / mst 
      
        return Np

    # Relaxation as defined by Spitzer. Here we consider the effect of mass spectrum due to BHs. 
    def _trh(self, M, rh, fbh, mbh, mst):
        """
        Calculate the relaxation timescale (`trh`) for a cluster, taking into account mass, radius, 
        BH fraction and time evolution.

        Parameters:
        -----------
        M : float
          Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass [Msun].
    
        mst : float
          Average stellar mass [Msun].

        Returns:
        --------
        float
           The relaxation timescale (`trh`) of the cluster, in [Myrs].

        Notes:
        ------
        - If the total mass `M` or the half-mass radius `rh` is less than or equal to zero, the function returns a very small value (`1e-99`). It is a precaution measure.
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass within `rh` is computed using a power-law fit based on the total mass and the number of particles.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, particle number `Np`
          and the gravitational constant `G`. The timescale is further modified by the cluster's `psi` value, which depends on the BH fraction.
        - If the cluster undergoes rigid rotation (`self.rotation` is True), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model. The expression by King is used.
        """
        
        if M <= 0 or rh <= 0: return 1e-99
        
        # Number of particles.
        Np = self._N(M, fbh, mbh, mst) 
        
        # Average mass within rh. We use a power-law fit.
        mav =  self.a3 * (M / Np) ** self.b3  # [Msun]
        
        # Relaxation.
        trh = 0.138 * sqrt(M * rh ** 3 / self.G) / (mav * self._psi(fbh, M, mbh, mst) * log(self.gamma * Np)) # [Myrs]. To avoid issues with the logarithm, we may add + numpy.e inside, but the current solutions stop when we have a few hundred stars.
        
        if (self.rotation): # Effect of rigid rotation.
            trh *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** (3 / 2) # Assume constant rotation for now.
        
        return trh
        
    # Relaxation for stars depending on the assumptions. 
    def _trhstar(self, M, rh, fbh, mbh, mst): 
        """
        Calculate the evaporation timescale (`trh`) for a cluster, taking into account mass and radius.

        Parameters:
        -----------
        M : float
          Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass [Msun].
    
        mst : float
          Average stellar mass [Msun].

        Returns:
        --------
        float
           The evaporation timescale (`trhstar`) of the cluster, in [Myrs].

        Notes:
        ------
        - If the total mass `M` or the half-mass radius `rh` is less than or equal to zero, the function returns a very small value (`1e-99`). It is a precaution measure.
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass within `rh` is computed using a power-law fit based on the total mass and the number of particles.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, particle number `Np`
          and the gravitational constant `G`. 
        - If one relaxation time scale is used (`self.two_relaxations` is False), the function is set equal to `trh`.
        - If the cluster undergoes rigid rotation (`self.rotation` is True), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model assuming constant rotation.
        """
        
        if M <= 0 or rh <= 0: return 1e-99
        
        # Half mass relaxation.
        trh = self._trh(M, rh, fbh, mbh, mst) # [Myrs]
         
        if not (self.two_relaxations): return trh # In case a model uses only one relaxation time scale.
        
        # Number of particles.
        Np = self._N(M, fbh, mbh, mst)
        
        # Use the average mass of the whole cluster to compute this relaxation timescale.
        mav = M / Np # [Msun]
        
       # Relaxation for evaporation.
        trhstar = 0.138 * sqrt(M * rh ** 3 / self.G) / (mav *  log(self.gamma * Np)) # [Myrs]
      
        if (self.rotation): # Effect of rigid rotation.
            trhstar *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** (3 / 2) # Assume constant rotation for now.
          
        return trhstar
          
     # Crossing time within the half-mass radius in Myrs.
    def _tcr(self, M, rh, tcross_index=None): 
        """
        Calculate the crossing timescale (`tcr`) for a cluster, which is related to the time it takes 
        for particles to cross the half-mass radius of the cluster.

        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].

        Returns:
        --------
        float
           The crossing timescale (`tcr`) of the cluster, in [Myrs].

        Notes:
        ------
        - If the total mass `M` or the half-mass radius `rh` is less than or equal to zero, the function returns a very small value (`1e-99`). It is a precaution measure.
        - The prefactor `k` is derived from the expression `tcr = 1 / sqrt(G * rhoh)`, where `rhoh` is the mass density within the half-mass radius.
          The value of `k` is adjusted by the `tcross_index` parameter, which can modify the crossing time based on specific cluster properties. An index is used in case the computation is performed at a different radius, for instance the tidal radius.
        - The crossing timescale is calculated using the formula: `tcr = k * sqrt(rh^3 / (G * M))`, where `rh` is the half-mass radius and `M` is the total mass of the cluster.
        """
        
        if M <= 0 or rh <= 0: return 1e-99
        
        # Prefactor derived from the expression tcr = 1 / sqrt(G rhoh). If the numerator is not 1, it needs to be changed trhough tcross_index.
        tcross_index = tcross_index or self.tcross_index
       
        k = 2 * sqrt(2 * pi / 3) * tcross_index
         
        tcr = k * sqrt(rh ** 3 / (self.G * M)) # [Myrs] If we have rotation, the numerial values of rh, M change and this impacts crossing time indirectly.
      
        return tcr
    
    # Escape velocity.
    def _vesc(self, M, rh):
        """
        Calculate the escape velocity (`vesc`) for a cluster, based on its total mass and half-mass radius.

        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius of the cluster [pc].

        Returns:
        --------
        float
           The escape velocity (`vesc`) at the center of the cluster, in [km/s].

        Notes:
        ------
        - The density `rhoh` is computed as the mass enclosed within the half-mass radius (`rh`), using the formula: 
          `rhoh = 3 * M / (8 * pi * rh ** 3)`. The density is in units of [Msun/pc^3].
        - The escape velocity `vesc` is then calculated using the relation: 
          `vesc = 50 * (M / 1e5) ** (1/3) * (rhoh / 1e5) ** (1/6)`, where the units for `vesc` are in [km/s]. 
          This formula expresses the central escape velocity based on the mass and the density of the cluster.
        - The escape velocity is further augmented by multiplying it by the factor `fc`, which adjusts the value according to the specific King model used for the cluster.
        - Time dependence if `fc` is neglected, however it should increase over time. This is introduced, currently as a trial, in a different function.
        """
        
        # Density.
        rhoh = 3 * M / (8 * pi * rh ** 3) # [Msun / pc^3]
        
        # Escape velocity.
        vesc = 50 * (M / 1e5) ** (1./3) * (rhoh / 1e5) ** (1./6) # [km/s] Central escape velocity as a function of mass and density. If it is needed in pc / Myrs, we multiply with 1.023.
        vesc *= self.fc # Augment the value for different King models. This in principle evolves with time but the constant value is an approximation.
       
        return vesc
        
    # Tides.
    def _xi(self, rh, rt):
        """
        Calculate the tidal parameter (`xi`) for a cluster, which describes the influence of tides
        based on the ratio of the half-mass radius to the tidal radius.

        Parameters:
        -----------
        rh : float
           Half-mass radius of the cluster [pc].
    
        rt : float
          Tidal radius of the cluster [pc].

        Returns:
        --------
        float
          The tidal parameter (`xi`), dimensionless.

        Notes:
        ------
        - The function calculates the tidal parameter using the respective expression from the dictionary. The default option is a power-law.
        """
        
        xi = self.tidal_models[self.tidal_model](rh, rt)
        
        return xi 
    
    def _beta_factor(self, S):
        """
        Retrieve the beta factor for BH ejection rate based on Spitzer's parameter.

        Parameters:
        -----------
        S : float
            Spitzer's parameter, which quantifies the mass segregation between BHs 
            and the surrounding stellar population. The value within rh is expected to be inserted.

        Returns:
        --------
        float
            The beta factor, which influences the ejection rate of BHs from the cluster.

        Notes:
        ------
        - The function uses a dictionary (`self.beta_dict`) to map different models (`self.beta_model`) 
          to corresponding beta factors as a function of Spitzer's parameter (`S`).
        - This approach allows for flexibility in defining how the beta factor is calculated, depending 
          on the chosen model in `self.beta_model`.
        - The beta factor is a critical parameter for describing the interaction and eventual ejection 
          of BHs in a cluster, especially as it relates to their segregation and dynamics. It is influenced indirectly by metallicity.
        """
        
        f = self.beta_dict[self.beta_model](S)
       
        return f

    
    # Varying stellar mass-loss rate.
    def _nu(self, Z, t):
        """
        Calculate the stellar mass-loss rate parameter (`nu`), which depends on metallicity (`Z`) and time (`t`),
        if these dependencies are activated.

        Parameters:
        -----------
        Z : float
           The metallicity of the cluster, typically as a ratio to the solar metallicity (e.g., Z/Zsolar).
    
        t : float
           Time elapsed since the beginning of the cluster's evolution [Myrs].

        Returns:
        --------
        float
           The mass-loss rate parameter (`nu`), which may vary with metallicity and time.

        Notes:
        ------
        - The function computes the mass-loss rate parameter `nu` using a default value (`self.nu`), and it introduces a time-dependent factor (`f`) 
          based on a quadratic logarithmic dependence on time. A future extension will introduce a dictionary so the user can select from the available dependences.
        - If the `sev_Z_run` flag is set to `True`, the mass-loss rate is further adjusted for metallicity (`Z`). This dependence allows for the effect of metallicity on mass loss, with `self.Zsolar` being the solar metallicity.
          Current observations do not show a significant dependence of the total stellar mass loss on metallicity. It will be studied in the future.
        - If the `sev_t_run` flag is set to `True` and the time-dependent factor `f` is positive, `nu` is multiplied by `f`, further modifying the mass-loss rate over time.
        """
        
        # Introduce a dependence of metallicity and time for stellar mass-loss rate. If activated.

        nu = self.sev_dict[self.sev_model](t) # Get the nu value from the dictionary.
        if nu < 0:
            nu = 0
        
        if (self.sev_Z_run):
            nu *= (Z / self.Zsolar) ** self.a # Another dependence on metallicity may be better.

        return nu
    
    def _fc(self, M, rh):
        
        """
        Estimate the change in parameter W0 and its impact on the central escape velocity
        
        Parameters:
        -----------
        M : float
           Total mass of the cluster [Msun].
    
        rh : float
           Half-mass radius [pc]
        
        Returns:
        --------
        float
           Multiplication factor for the central escape velocity. In principle, this parameter scales with sqrt(W0).
        
        """
        
        fc = (self.M0 / M) ** self.d1 * (rh / self.rh0) ** self.d2 # Such parameter should increase because the central region becomes denser over time. Will be studied in the future.
        
        return fc
    
    # Rate of change of the average stellar mass due to stellar evolution.
    def _mstdot_sev(self, mst, t):
        """
        Calculate the rate of change of average stellar mass (`mstdot`) due to mass loss, 
        as a function of the stellar mass (`mst`) and time (`t`).

        Parameters:
        -----------
        mst : float
            Average stellar mass of the cluster at time `t` [Msun].
    
        t : float
            Time elapsed since the beginning of the cluster's evolution [Myrs].

        Returns:
        --------
        float
            The rate of change of the average stellar mass (`mstdot`), calculated as a negative value 
            indicating mass loss.

        Notes:
        ------
        - The mass-loss rate is proportional to the stellar mass `mst` and inversely proportional to time `t`.
        - The proportionality factor is given by `nu(self.Z, t)`, which accounts for the metallicity and time 
          dependencies of the mass-loss rate should the user select to include any dependence.
        - A negative value for `mstdot` is returned to indicate mass loss.
        - The function assumes that `nu` properly handles cases where time or metallicity could result in 
          unphysical behavior.
        """
       
        mstdot = - self._nu(self.Z, t) * mst / t # [Msun / Myrs]
       
        return mstdot
    
    def _mstdot_ev(self, mst, xi, trh):
        """
        Calculate the rate of change of average stellar mass (`mstdot`) due to tidal mass loss, 
        as a function of the stellar mass (`mst`), evaporation rate (`xi`) and relaxation (`trh`).

        Parameters:
        -----------
        mst : float
            Average stellar mass of the cluster at time `t` [Msun].
    
        trh : float
            Relaxation time scale for evaporation [Myrs]. Relaxation for evaporation should be used.

        Returns:
        --------
        float
            The rate of change of the average stellar mass (`mstdot`), calculated as a positive value 
            indicating mass increase. This is because the lightest stars are removed from the cluster due to tides.

        Notes:
        ------
        - The mass-loss rate is proportional to the stellar mass `mst`, evaporation rate `xi` and inversely proportional to relaxation `trh`.
        - A positive value for `mstdot` is returned to indicate increase. Tides rip off light stars.
        """
        # Constrain mst_dot to be above the lowest possible mass, and below the mass of remnants.
        mstdot =  self.chi * (1 - self.m_break1 / mst) * (1 - mst / self.mst_inf) * xi * mst / trh # [Msun / Myrs] Currently in trial mode.
      
        return mstdot
    
    # Smoothing function used to connect the unbalanced phase with the balanced.
    def _balance(self, t):
        """
        Determine a smoothing factor for transitioning between the unbalanced phase (prior to core collapse)
        and the balanced phase (post core collapse) of a cluster.

        Parameters:
        -----------
        t : float
            Time in [Myrs].

        Returns:
        --------
        float
            A smoothing factor, typically between 0 and 1, where:
            - 0 corresponds to the start of unbalanced phase.
            - 1 corresponds to the balanced phase.
            - Intermediate values represent a smooth transition between these phases.

        Notes:
        ------
        - The smoothing factor is determined by applying a predefined function from `self.balance_dict`,
          which contains models for the transition.
        - The specific model used is selected by `self.balance_model`.
        - This function allows flexibility to test different smoothing models depending on the desired behavior.
          It is used such that the differential equations to be solved become continuous at t=tcc. A similar function, but more steep, may be used fot tsev as well.

        """
        z = self.balance_dict[self.balance_model](t)
      
        return z
    
    # We construct the differential equations to be solved. 
    def odes(self, t, y):
        """
        Compute the time derivatives of the system's state variables, which describe the evolution of a star cluster 
        under various physical processes including stellar evolution, tidal effects, mass segregation, and core collapse.

        Parameters:
        -----------
        t : float
           Time elapsed since the beginning of the cluster's evolution, in [Myrs].
    
        y : array-like
           A sequence representing the state variables:
           - y[0] : float : Stellar mass, `Mst` [Msun].
           - y[1] : float : Black hole mass, `Mbh` [Msun].
           - y[2] : float : Half-mass radius, `rh` [pc].
           - y[3] : float : Parameter describing mass segregation, `Mval`.
           - y[4] : float : Parameter describing the ratio `rh / rv`.
           - y[5] : float : Average stellar mass `mst` [Msun].

        Returns:
        --------
        numpy.array
           An array containing the time derivatives of the state variables:
           - `Mst_dot`: Stellar mass loss rate [Msun/Myr].
           - `Mbh_dot`: Black hole mass loss rate [Msun/Myr].
           - `rh_dot` : Rate of change of the half-mass radius [pc/Myr].
           - `Mval_dot` : Evolution of the mass segregation parameter [1/Myr].
           - `r_dot` : Evolution of half-mass over Virial radius [1/Myr]
           - `mst_dot`: Average stellar mass loss rate [Msun/Myr]

        Notes:
        ------
        - This function models the dynamical evolution of the star cluster using multiple physical processes.
        - The calculations include:
        1. **Core Collapse**:
         - After core collapse time (`tcc`), the parameter for mass segregation, `Mval`, transitions to a constant value. In the future, where a smoothing transition will be inserted, this contribution will not be needed.
        2. **Stellar Evolution**:
         - Stellar mass loss due to stellar evolution is modeled using mass-loss rate (`nu`). Function `_nu` is used.
         - Mass segregation is evolved if the option is selected.
         - Virial radius evolves differently in the unbalanced phase if the option is selected. In the future the smoothing function will be included.
         - The average stellar mass also evolves due to sev. It accounts the proper solution, if a running ν is selected.
        3. **Tidal Effects**:
         - If tides are active, they contribute to mass loss and affect the half-mass radius evolution.
         - Finite escape time and rotation are also included in the tidal mass-loss calculation if selected.
         - In the future, the impact of tides on the average stellar mass will be included. Should affect low galactocentric distances and light star clusters mainly.
        4. **BH Mass Loss**:
         - BH mass loss includes mechanisms for ejection based on cluster parameters and equipartition states. Tidal effects are also available.
        5. **Cluster Expansion**:
         - Expansion due to energy loss is modeled using Hénon's parameter (`zeta`). A constant value is assumed. In the future, a smoother connection will be made.
         - The contribution of evaporating stars to half-mass radius changes is also included, if the option is activated.
        6. **Combined Mass Loss**:
         - Total mass loss and its impact on the cluster size are calculated for both stars and BHs.
      
        - The returned derivatives (`Mst_dot`, `Mbh_dot`, `rh_dot`, `Mval_dot`) can be used in numerical integrators to evolve the cluster state over time.
        """
        
        Mst = y[0] # [Msun] Mass of stars.
        Mbh = y[1] # [Mun] Mass of BHs.
        rh = y[2] # [pc] Half-mass radius.
        Mval = y[3] # Parameter for mass segregation.
        r = y[4] # Ratio rh / rv.
        mst = y[5] # [Msun] Average stellar mass.
                
        tcc = self.tcc  # [Myrs] Core collapse.
        tsev = self.tsev # [Myrs] Stellar evolution.
        tbh = self.t_bhcreation # [Myrs] Time instance when BHs have been created.

        mbh = self._mbh(Mbh)  # [Msun] Extract the average BH mass.
        if Mbh < mbh: Mbh, mbh = 0, 1e-99 # It would be unphysical to have a BH mass that is lesser than the average BH mass. This happens only for the final BH of course.
        if Mst < self.mst_inf : Mst, mst = 0, 1e-99 # In case the total stellar mass drops below the mass of a single remnant, it effectively means that all stars have been ejected.
       # F = self._balance(t)
        
        M = Mst + Mbh # [Msun] Total mass of the cluster. It overestimates a bit initially because we assume Mbh > 0 from the start.
        fbh = Mbh / M # Fraction of BHs. Before core collapse is it overestimated
        S = self.a11 * self.a12 ** (3 / 2) * (Mbh / Mst) ** self.b1 * (mbh / mst) ** (3 / 2 * self.b2) # Spitzer's parameter for equipartition.
        if S < self.S_crit: # We keep this constant when we reach equipartition so that the ejection rate remains fixed.
            S = self.Sf # Constant value when we have complete equipartition. Should be equal to or below the critical value for equipartition.
       
        beta_f =  self._beta_factor(S) # Factor used to modulate BH and stellar ejections. Depends on Spitzer´s parameter, and becomes constant when we reach equipartition.
        Np = self._N(M, fbh, mbh, t)  # Total number of particles.
    #    m = M / Np # [Msun] Total average mass.
        
        rt = self._rt(M) # [pc] Tidal radius.
        trh = self._trh(M, rh, fbh, mbh, mst) # [Myrs] Relaxation.
        trhstar = self._trhstar(M, rh, fbh, mbh, mst) # [Myrs] Evaporation time scale.
        tcr = self._tcr(M, rh) # [Myrs] Crossing time.
      
        xi = 0 # Tides are turned off initially. We build them up if needed.
        alpha_c = 0 # Ejection rate of stars from the center are set equal to zero.
        
        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0 # [Msun / Myrs], [pc / Myrs], [Msun / Myrs] At first the derivatives are set equal to zero, then we build them up.
        Mval_dot = 0 # [1 / Myrs] Derivative for parameter of mass segregation.
        r_dot = 0 # [1 / Myrs] Derivative for the ratio rh / rv.
        mst_dot = 0 # [Msun / Myrs] Derivative for the average stellar mass.
       
        M_val = Mval # This parameter describes mass segregation, used in the equation for rh. The reason why it changes to a constant value after core collapse is because of Henon's statement, here described by parameter zeta.
        if t >= tcc: # We may neglect this statement if F is present.
            M_val = self.Mval_cc # After core collapse, a constant value is assumed, different. In the isolated version of clusterBH the value is 1. For Mval_cc = 2, no contribution is assumed.
                  
        Mst_dotsev = 0 # [Msun / Myrs] Mass loss rate from stellar evolution.
               
       # Stellar mass loss.
        
        if t >= tsev and Mst > 0: # This contribution is present only when Mst is nonzero.
            nu = self._nu(self.Z, t) # Rate of stellar mass loss due to stellar winds.
            mst_dot += self._mstdot_sev(mst, t) # [Msun/Myrs] When we consider stellar evolution, the average stellar mass changes through this differential equation. It is selected so that the case of a varying nu is properly described, if activated.
            Mst_dotsev -= nu * Mst / t # [Msun / Myrs] Stars lose mass due to stellar evolution.
            rh_dot -= (M_val - 2) *  Mst_dotsev / M * rh # [pc / Myrs] The cluster expands for this reason. It is because we assume a uniform distribution initially. 
        
        # Check for the impact of the Virial radius on the half-mass expansion.
        if t < tcc and (self.Virial_evolution):
           r_dot += r * (self.rf - r) / trh # * F # [1 / Myrs] Because the Virial radius evolves slower compared to the half-mass radius, we account for this difference. Does not evolve in the balanced phase.
           rh_dot += rh * r_dot / r # Correction term from energy variation. A better expression for r_dot is needed.
       
        # Check if the cluster segregates.
        if t < tcc and (self.mass_segregation):
           Mval_dot += Mval * (self.Mvalf - Mval) / trh  # [1 / Myrs] Evolution of parameter describing mass segregation. 
       
        # Add tidal mass loss.
        
        if (self.tidal): # Check if we have tides.  
            
           xi += self._xi(rh, rt) # Tides. With Fm it should be xi += (self.f + F * (1 - self.f)) * self._xi(rh, rt)
        
           if (self.finite_escape_time): 
              P = self.p * (trhstar / tcr) ** (1 - self.x)  # Check if we have finite escape time from the Lagrange points.
              xi *= P
              
           if (self.rotation): # If the cluster rotates, the evaporation rate changes. A similar change to relaxation is used.
               xi *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** self.gamma1
        
        # Check for stellar ejections.
        if t >= tcc and Mst > 0: # A proper description should start with zero alpha_c, increase slightly up until core collapse, be a function of fbh, mbh and when we have no BHs it maximises.
           
           alpha_c += self.alpha_ci # Initial ejection rate of stars. With F, it should be alpha_c += self.alpha_ci * F 
           
           if (self.running_bh_ejection_rate_2):
               alpha_c += (self.alpha_cf - self.alpha_ci) * (1 - beta_f) # * F # This expression states that when we have many BHs, the ejection rate of stars may be small, and when the BHs vanish it increases. We use a similar rate of change as for the BH ejection rate. 
           
           if (self.running_stellar_ejection_rate):
               alpha_c *= (1 - 5 * xi / (3 * self.zeta)) # We may alter alpha_c based on tides as introduced in EMACSS.
               
      # Expansion due to energy loss. It is due to the fact that the cluster has a negative heat capacity due to its gravitational nature.
        if t >= tcc:
           rh_dot += self.zeta * rh / trh # [pc / Myrs] # Instead of the if statement, simply multiplying with F should work.
        
        if Mst > 0:  # Tidal mass loss and stellar ejections appears only when we have stars.                 
           mst_dot += self._mstdot_ev(mst, xi, trhstar) # [Msun / Myrs] Rate of change of average stellar mass due to tides.  
           Mst_dot -= xi * Mst / trhstar + alpha_c * self.zeta * M / trh # [Msun / Myrs] Mass loss rate of stars due to tides (and central ejections).
           rh_dot += 2 * Mst_dot / M * rh # [pc / Myrs] Impact of tidal mass loss to the size of the cluster.
        
        Mst_dot += Mst_dotsev  # [Msun / Myrs] Correct the total stellar mass loss rate.
   
        # Effect of evaporating stars on the half mass radius. Keep only evaporation, not ejections here so use xi and not xi_total.   
        if (self.escapers) and (self.tidal): # It is connected to the model for the potential of the cluster. 
           index = self.cluster_model_dict[self.cluster_model](rt, M) # [pc^2 / Myrs^2 ] Get the value from the dictionary. It changes with respect to the potential assumed for the cluster.
           index = abs(index) / (self. G * M) # Prefactor for energy contribution, should scale as 1 / [pc].
           kin = self.kin # Default value is constant. Should the user wish to vary it, they can select it in the following line.

           if (self.varying_kin): # If activated, it introduces a dependence on the number of particles, changing with time.
               kin =  2 * (self._tcr(M, rh, tcross_index= 1 / (2 * sqrt(2 * pi / 3))) * 0.138 / trhstar) ** (1 - self.x)  # Energy based criterion similar to factor P in finite escape time. They should be together. The constants in front are irrelevant.
 
           rh_dot += 6 * xi / r / trhstar * (1 - kin) * rh ** 2 * index * Mst / M # [pc / Myrs] If escapers carry negative energy as they leave, the half mass radius is expected to increase since it is similar to emitting positive energy. The effect is proportional to tides.
           # The above expression checks the potential used for a globular cluster. In the end, it uses φ_tidal = 3 / 2 φ(rt) 
           
        if Mbh > 0 and t >= tcc: # Check if we have BHs so that we can evolve them as well.
              
           beta = self.beta # Ejection rate of BHs. Initially it is set as a constant. It should be changed when fbh increases a lot and the light component does not dominate anymore.
        
           if fbh < self.fbh_crit and (self.running_bh_ejection_rate_1): # Condition for decreasing the ejection rate of BHs due to E / M φ0. # A condition for mbh_crit may be needed.
               beta *=  (fbh / self.fbh_crit) ** self.b4 #* (mbh / m / self.qbh_crit) ** self.b5 # A dependence on average mass (or metallicity) may be needed.
           
           if (self.running_bh_ejection_rate_2): # Decrease the ejection rate for clusters that are close to reaching equipartition.
              beta *= beta_f # Another ansatz with S-dependence may be chosen here. However, for large S we should have a constant beta and for vanishing S, beta vanishes as well.
           
           Mbh_dot -= beta * self.zeta * M / trh  # [Msun / Myrs] Ejection of BHs each relaxation. 
       
        # If the tidal field was important, an additional mass-loss mechanism would be needed.
        if (self.dark_clusters) and Mbh > 0 and t >= tbh: # Appears when BHs have been created.
           gbh = (exp(self.c_bh * fbh) - 1) / (exp(self.c_bh) - 1) # A simple dependence on fbh is shown.
           xi_bh = xi * gbh # Modify the tidal field that BHs feel such that it is negligible when we have many stars, and it starts increasing when we have a few only. 
           Mbh_dot -= xi_bh * Mbh / trhstar # [Msun / Myrs] Same description as stellar mass loss. Use relaxation in the outskirts.
          # mbh should increase with the tidal field. Another prescription is needed for mbh. 
          
           # Now if this is important and escapers is activated, it should be included here
           if (self.escapers): # Currently it is not applied to BHs if the tidal field affects them.
              rh_dot += 6 * xi_bh / r / trhstar * (1 - kin) * rh ** 2 * index * Mbh / M # [pc / Myrs] 
                  
        rh_dot += 2 * Mbh_dot / M * rh # [pc / Myrs] Contraction since BHs are removed.
    
        derivs = numpy.array([Mst_dot, Mbh_dot, rh_dot, Mval_dot, r_dot, mst_dot], dtype=object) # Save all derivatives in a sequence.

        return derivs # Return the derivatives in an array.

   # Extract the solution using the above differential equations.
    def evolve(self, N, rhoh):
        """
        Simulates the dynamical evolution of a star cluster over time using the specified initial conditions and physical models.

        Parameters:
        -----------
        N : int
          Total number of particles (stars and black holes) in the cluster.
        rhoh : float
          Initial half-mass density of the cluster [Msun/pc^3].

        Method:
        -------
        - Uses `solve_ivp` to numerically integrate the system's differential equations (`odes` method).
        - Implements an event to terminate the simulation when the stellar mass drops below a minimum threshold (`Mst_min`) and the BH population is also below a threshold (`Mbh_min`).

        Output:
        ------
        - Cluster parameters are computed and can be saved in a text file if selected.
        
        Notes:
        ------
        - Initial conditions for stellar mass, BH mass, half-mass radius, segregation parameter, ratio `rh/rv` and average stellar mass are set to 
          `self.M0`, `self.Mbh0`, `self.rh0`, `self.Mval0`, `self.r0`, `self.m0` respectively.
        - Results are stored as attributes of the object for further analysis.
        """
        
        Mst = [self.M0] # [Msun] Initial stellar mass.
        Mbh = [self.Mbh0] # [Msun] Initial BH mass.
        rh = [self.rh0] # [pc] Initial half-mass radius.
        Mval = [self.Mval0] # Initial parameter for mass segregation.
        r = [self.r0] # Initial value for rh / rv.
        mst = [self.m0] # [Msun] Initial average stellar mass.
       
        y = [Mst[0], Mbh[0], rh[0], Mval[0], r[0], mst[0]] # Combine them in a multivariable.

        def Mst_min_event(t, y):  # Event in order to stop when stars are lost. Assumes BHs have been ejected first.
            return y[0] - self.Mst_min
        
        def Mbh_min_event(t, y): # Event in order to stop when BHs are lost. Assumes stars have been evaporated first.
            return y[1] - self.Mbh_min
        
        def combined_event(t, y):
            # Check if either condition is still valid
            condition_1 = Mst_min_event(t, y)  # Positive if stars are above the threshold
            condition_2 = Mbh_min_event(t, y)  # Positive if BHs are above the threshold
 
            # Event function returns the maximum of the two conditions. Stops only when both are negative.
            return max(condition_1, condition_2) # Maximum is chosen because it suggests that this population dominates.

        # Configure the event properties.
        combined_event.terminal = True  # Stop integration when the event is triggered.
        combined_event.direction = -1    # Trigger regardless of whether the function increases or decreases.

        t_eval = numpy.arange(0, self.tend, self.dtout) if self.dtout is not None else None # [Myrs] Time steps to which the solution will be computed.
        
        # Solution.
        sol = solve_ivp(self.odes, [0, self.tend], y, method=self.integration_method, t_eval=t_eval, events=[combined_event], rtol=1e-8, atol=1e-10) 

        self.t = numpy.array([x / 1e3 for x in sol.t]) # [Gyrs] Time.
        self.Mst = sol.y[0] # [Msun] Stellar mass.
        self.Mbh = sol.y[1] # [Msun] BH mass.
        self.rh = sol.y[2] # [pc] Half-mass radius
        self.Mval = sol.y[3] # Parameter for segregation
        self.r = sol.y[4] # Ratio rh / rv
        self.mst = sol.y[5] # [Msun] Average stellar mass.
       
        self.mbh = numpy.array([self._mbh(x) if y > self.tcc else self.mbh0 for (x, y) in zip(self.Mbh, sol.t)]) # [Msun] Average BH mass.
        self.Mbh = numpy.array([x if x >= y else 0 for x, y in zip(self.Mbh, self.mbh)]) # [Msun] BH mass corrected for mbh > Mbh.
        self.mbh = numpy.array([y if x >= y else 1e-99 for x, y in zip(self.Mbh, self.mbh)]) # [Msun] Correct the average BH mass.
        
        # Quantities for the cluster.
        self.M = self.Mst + self.Mbh # [Msun] Total mass of the cluster. We include BHs already.
        self.rt = self._rt(self.M) # [pc] Tidal radius.
        self.fbh = self.Mbh / self.M # BH fraction.
        self.rv = self.rh / self.r # [pc] Virial radius.
        self.psi = numpy.array([self._psi(x, y, z, u) for (x, y, z, u) in zip(self.fbh, self.M, self.mbh, self.mst)]) # Friction term ψ.
        self.Np = numpy.array([self._N(x, y, z, u) for (x, y, z, u) in zip(self.M, self.fbh, self.mbh, self.mst)]) # Number of components. 
        self.mav = self.M / self.Np # [Msun] Average mass of cluster over time, includes BHs. No significant change is expected given that Nbh <= O(1e3), apart from the beginning where the difference is a few percent.
        self.mst_sev = self._mst_sev() # [Msun] Average stellar mass, considering only stellar evolution.
        self.Nbh = self.Mbh / self.mbh # Number of BHs.
        self.E = - self.r * self.G * self.M ** 2 / (4 * self.rh) # [pc^2 Msun / Myrs^2]
        if (self.tidal):
            self.xi = self._xi(self.rh, self.rt) # Evaporation rate.
        self.trh = numpy.array([self._trh(x, y, z, u, v) for x, y, z, u, v in zip(self.M, self.rh, self.fbh, self.mbh, self.mst)]) # [Myrs] Relaxation within rh.
        self.trhstar = numpy.array([self._trhstar(x, y, z, u, v) for x, y, z, u, v in zip(self.M, self.rh, self.fbh, self.mbh, self.mst)]) # [Myrs] Relaxation within rh.
        self.tcr = numpy.array([self._tcr(x, y) for x, y in zip(self.M, self.rh)]) # [Myrs] Crossing time.
        self.vesc = numpy.array([self._vesc(x, y) for x, y in zip(self.M, self.rh)]) # [km/s] Escape velocity. 
        self.S = self.a11 * self.a12 ** (3 / 2) * (self.Mbh / self.Mst) ** self.b1 * (self.mbh / self.mst) ** (3 / 2 * self.b2) # Parameter indicative of equipartition.
        self.phi_c = self.cluster_model_dict[self.cluster_model](self.rt, self.M) # [pc^2/Myr^2] Cluster potential at the tidal radius.
        self.phi_0 = - (1.023 * self.vesc) ** 2 / 2 # [pc^2/Myr^2] Central potential. Needs fc to be correct.
        self.fcrun = numpy.array([self._fc(x, y) for (x, y) in zip(self.M, self.rh)]) # Parameter to estimate how the escape velocity evolves.
        self.beta_run = self.beta * self._beta_factor(self.S) # Running beta factor.
        self.nu_run = self._nu(self.Z, sol.t) # Running stellar mass loss rate.
        
        self.mbh_max = numpy.array([self._find_mmax(x) for x in self.Mbh]) # [Msun] Maximum BH mass in the IMF as a function of time. Increasing the number of bins make it more accurate.
        # Check if we save results. The default option is to save the solutions of the differential equations as well as the tidal radius and average masses of the two components. Additional inclusions are possible.
        if self.output:
           with open(self.outfile, "w") as f:
        # Header
              f.write("# t[Gyrs]    Mbh[msun]   Mst[msun]     rh[pc]     rt[pc]     mbh[msun]   mst[msun]   mbh_max[msun]\n")
        
        # Data rows
              for i in range(len(self.t)):
                  f.write("%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e\n" % (
                self.t[i], self.Mbh[i], self.Mst[i], self.rh[i], self.rt[i],
                self.mbh[i], self.mst_sev[i], self.mbh_max[i]
                ))


#######################################################################################################################################################################################################################################################
"""
Notes:
- Another set of values that can be used is:
    zeta, beta, n, Rht, ntrh, b, Mval0, Mval_cc, S0 = 0.062, 0.077, 0.866, 0.046, 0.263, 2.2, 3.119, 3.855, 1.527
    zeta, beta, n, Rht, ntrh, b, Mval0, Mval_cc, S0 = 0.0977, 0.0566, 0.7113, 0.0633, 0.1665, 2.14, 3.1156, 3.1156, 1.96.

- Functions _mstdot_ev, _balance and _fc are works in progress, currently deactivated or unused.
- self.Virial_evolution, self.finite_escape_time, self.rotation, self.dark_clusters and self.sev_Z_run are future statemens that require refinement.

Use:
- An isolated cluster can be described with tidal=False. It can have stellar ejections or not.
- Simple cases where we have no stellar evolution can be described with nu=0.
- If we start with BHs in the balanced phase at t=0, use ntrh=0.
- If no change in the BH ejection rate is wanted, use a value for S0 quite close to 0 or disable running_bh_ejection_rate.
- A richer IMF can be specified by assigning values to additional mass breaks for the intervals, slopes and bins. Default option is 3. They must be inserted in sequence, so m_break5, a_slope4, nbin4 are the next inclusions. Masses must increase. 

Limitations:
- Dissolved clusters cannot be described. A varying mst is needed, and tidal effects must be considered in the stellar average mass.
- A more accurate connection between the unbalanced and the balanced phase may be needed. See EMACSS.
- Description of dark clusters has not been studied extensively. 
- Rotation, if activated, needs to be more accurate since now it remains constant.
- The special case of RV-filling clusters does not consider stellar induced mass loss rate. It can be inserted however in the odes function.
- Tidal shocks for interactions with GMC's for instance are not considered here. Can be included in the future.

Room for improvement:
- When self.dark_clusters is True, and fbh increases, the time scale for evaporation may need to be reconsidered, the evaporation rate could use a different expression. The ejection rate of BHs however, should be changed, it should not follow this law above a given fraction. The average BH mass should in turn increase due to evaporation.
- When self.Virial_evolution is considered, the current version is not exactly accurate. A better, smoothing function needs to be used.
- When self.rotation is used, the model considers rigid rotation and the same impact on the evaporation rate. This may not be true necessarily, and a more physical model for rotation may be needed.

"""



########################################################################################################################################################################################################################################################
