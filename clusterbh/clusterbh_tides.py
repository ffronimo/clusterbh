from __future__ import division
import numpy
from numpy import log, sqrt, pi, log10, exp, tanh
from scipy.special import erf, hyp2f1
from scipy.integrate import solve_ivp, cumulative_trapezoid
numpy.seterr(invalid='ignore', divide='ignore')

"""
- The current parameters only work with the default settings. If a different model is chosen, a different set of values may be needed.
- The default option requires the Simple Stellar Population package. To install it, run 'pip install astro-ssptools==2.0.1'. Version 2.0.1 uses Zsolar=0.02. If not, simply state ssp=False in kwargs before a run.
  For more details, visit SMU-clusters/ssptools. 
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
        self.m0 = None # [Msun] Initial average mass. If unspecified, it is computed directly from the IMF. The computation is skipped if the user inserts it as argument.
        self.fc = 1 # Factor for escape velocity. It is different for other King models. By default, W0 is set to 5 (normalized) and scales roughly as sqrt(W0/5).
        self.rg = 8 # [kpc] Galactocentric distance of the cluster. For eccentric orbits, consider it as a(1 - e) where a is the semi-major axis, e is eccentricity.
        self.Z = 0.0002 # Metalicity of the cluster. Default to low-metallicity. 
        self.omega0 = 0 # [1 / Myrs] Angular frequency of cluster due to rotation.
        
        # BHMF. It is needed only if the user does not use the available Simple Stellar Population (SSP) tools.
        # The current treatment for the BHMF is a simple power-law. At each iteration after core collapse, the BHMF is updated. The upper mass decreases with time, as dictated by Mbh.
        self.f0 = 0.06 # Initial fraction of BHs. It decreases with metallicity and is important only when the SSP are not used.
        self.mlo = 3 # [Msun] Lowest BH mass in the BHMF.
        self.mup = 30 # [Msun] Upper BH mass in the BHMF
        self.mb = 0. # [Msun] Lowest mass unaffected by kicks. Default option is for no kicks. If kicks are activated, it is computed subsequently.
        self.alpha_BH = 0.5 # Slope in the BHMF. If kicks are considered, the slope cannot be equal to -5 - 3k, where k is a positive integer. The special value of -2 (k=-1) is considered, but other values are not. These can be used, but without kicks.
        self.fretm = 1 # Retention fraction of BHs. The default value neglects kicks. Changes the initial BH mass only trhough the prescription implemented in this script.
        self.t_bhcreation = 8 # [Myrs] Time required to create all BHs.
        self.N_points = 500 # Number of points used for even spacing.
        
        # Model parameters. 
        self.mns = 1.4 # [Msun] Mass of Neutron Stars (NS).
        self.mst_inf = 1.4 # [Msun] The maximum upper mass for stars at infinity. It represents the mass of the remnants, typically near the limit for the heaviest white dwarf or lightest neutron star. It serves as the upper boundary for the average stellar mass.
        self.sigmans = 265 # [km/s] Velocity dispersion for NS. 
        self.gamma = 0.02 # Parameter for Coulomb logarithm.
        self.S_crit = 0.0 # Critical value for Spitzer's parameter, below which the system reaches complete equipartition. Currently deactivated. 
        self.Sf = 0.0 # Auxiliary value for Spitzer's parameter when equipartition is reached. It ensures the BH ejection rate stays finite and nonzero. It should be equal to or smaller than the critical value where full equipartition occurs. Currently deactivated.
        self.x = 3./4 # Exponent used for finite escape time from Lagrange points.
        self.r0 = 0.8 # Ratio of the half-mass radius over the Virial radius initially.
        self.rf = 1 # Final value for ratio rh / rv. Used only if the option for the evolution of rh / rv with time is selected. 
        self.tcross_index = 2 * sqrt(2 * pi / 3) # Prefactor for half-mass crossing time defined as tcr = index/sqrt(G ρ).
        self.c = 10 # Parameter for exponential tides. Can be used for other tidal models available.
        self.c_bh = 10 # Exponent for tides used in the BH evaporation rate. Currently deactivated. Used for describing the impact of the tidal field on the BH population. 
        self.rp = 0.8 # [kpc] Constant parameter for galactic potentials. It serves as an effective scale.
        self.rpc = 1 # [pc] Constant parameter for cluster potentials, for instance Plummer. Also an effective scale.
        self.fmin = 0 # Constant evaporation rate. Can be used in order to have a constant evaporation rate for cases where the ratio rh / rt is not strong enough. Currently deactivated.
        
        # This part includes parameters that are on trial, intended for future additions. Some may be subjects to future fittings.
        #_____________________________________________
        self.gamma1 = 3 / 2 # Exponent for the evaporation rate in the rotational component. Default value keeps the ratio of xi to trh constant, with changes affecting only the evolution of rh.
        self.gamma2 = 1 # Parameter used for describing the BH ejection rate. Applies to extended options.
        self.gamma3 = 3 # Constant used in the balance function. Currently deactivated.
        self.gamma4 = 2 # Second constant for the balance function, working with gamma3 to smoothly connect the two phases.
        self.c1 = 0 # Parameter for stellar evolution. Can be used for a time dependent ν.
        self.c2 = 0 # Second parameter for stellar evolution. 
        self.d1 = 0.18 # First exponent to describe the evolution of parameter fc. Can be used in order to specify how W0 evolves for a given cluster as function fo M and rh.
        self.d2 = 0.14 # Second exponent to describe the evolution of parameter fc.
        self.chi = 0. # Prefactor that adjusts the average stellar mass with respect to tides, particularly important for small galactocentric distances.
        self.Rht_crit = 0.3 # Critical value of ratio rh / rt above which induced mass-loss from stellar evolution should be considered.
        #_________________________________________________
        
        # Parameters fitted to N-body / Monte Carlo models.
        self.zeta = 0.085 # Energy loss per half-mass relaxation. It is assumed to be constant.
        self.beta = 0.0638 # BH ejection rate from the core per relaxation, considered constant for all clusters, assuming a large BH population in the core, regardless of initial conditions. Shows the efficiency of ejections. Subjects to change in the following functions.
        self.nu = 0.072 # Mass loss rate of stars due to stellar evolution. It is the same for all clusters and does not depend on metallicity. Changes can be applied in the following functions.
        self.tsev = 1 # [Myrs]. Time instance when stars start evolving. Typically it should be a few Myrs.
        self.a0 = 1 # Fix zeroth order in ψ, if approximations are made. In reality, it serves as the contribution of the stellar population to the half-mass mass spectrum.
        self.a11 = 1.94 # First prefactor in ψ. Relates the BH mass fraction within the half-mass to the total fraction.
        self.a12 = 0.634 # Second prefactor in ψ. Relates the mass ratio of BHs over the total average mass, within the half-mass radius and the total. It is the product of a11 and a12 that matters in the following computations.
        self.a3 = 0.984 # [Msun] Prefactor of average mass within the half-mass radius compared to the total. The values of a11, a12 and a3 are taken for clusters with large initial relaxations.
        self.n = 1.6222 # Exponent in the power-law parametrization of tides. It determines how quickly the Virial radii lose mass. Its value relative to 1.5 indicates the rate at which mass is lost.
        self.Rht = 0.113 # Ratio of rh/rt, used to calculate the correct mass loss rate due to tides. It serves as a reference scale, not necessarily representing the final value of rh/rt.
        self.ntrh = 1.0967 # Number of initial relaxations to compute core collapse instance. Marks also the transition from the unbalanced to the balanced phase.
        self.alpha_ci = 0.0 # Initial ejection rate of stars, when BHs are present. Currently deactivated, as it only accounts for a small percentage of the total stellar mass loss each relaxation.
        self.alpha_cf = 0.0 # Final ejection rate of stars, when all BHs have been ejected (or the final binary remains) and the stellar density increases in the core. Represented by different values, but in principle they may be equal.
        self.kin = 1 # Kinetic term of evaporating stars. Used to compute the energy carried out by evaporated stars. The deafult value neglects this contribution. The case of a varying value is available in later parts.
        self.b = 2.14 # Exponent for parameter ψ. The choices are between 2 and 2.5, the former indicating equal velocity dispersion between components while the latter complete equipartition.
        self.b_min = 2 # Minimum exponent for parameter ψ. Can be used when the exponent in ψ is varying with the BH fraction.
        self.b_max = 2.5 # Maximum exponent for parameter ψ. Taken from Wang (2020), can be used for the case of a running exponent b.
        self.b0 = 2 # Exponent for parameter ψ. It relates the fraction mst / m (average stelar mass over average mass) within rh and the total fraction.
        self.b1 = 1 # Correction to exponent of fbh in parameter ψ. It appears because the properties within the half-mass radius differ.
        self.b2 = 1.04 # Exponent of fraction mbh / m (average BH mass over average mass) in ψ. Connects properties within rh with the global. Current value is used for sparse clusters.
        self.b3 = 0.96 # Exponent of average mass within rh compared to the global average mass. All exponents are considered for the case of clusters with large relaxation.
        self.b4 = 0.17 # Exponent for the BH ejection rate. Participates after a critical value, here denoted as fbh_crit.
        self.b5 = 0.4 # Second exponent for the BH ejection rate. Participates after a critical value for the BH fraction, here denoted as qbh_crit. It is deactivated for now.
        self.Mval0 = 3.1187 # Initial value for mass segregation. For a homologous distribution of stars, set it equal to 3.
        self.Mvalf = 3.1187 # Final parameter for mass segregation. Describes the maximum value for the level of segregation. It is universal for all clusters and appears at the moment of core collapse.
        self.p = 0.1 # Parameter for finite time stellar evaporation from the Lagrange points. Relates escape time with relaxation and crossing time. 
        self.fbh_crit = 0.005 # Critical value of the BH fraction to use in the ejection rate of BHs. Decreases the fractions E / M φ0 properly, which for BH fractions close/above O(1)% approximately can be treated as constant.
        self.qbh_crit = 25 # Ratio of mbh / m when the BH ejection rate starts decreasing. Also for the ratio E / M φ0, however now it is deactivated. Should introduce an effective metallicity dependence.
        self.S0 = 1.7247 # Parameter used for describing BH ejections when close to equipartition. Useful for describing clusters with different metallicities using the same set of parameters, as well as for clusters with small BH populations which inevitably slow down the ejections to some extend.
        self.gamma_exp = 10 # Parameter used for obtaining the correct exponent for parameter ψ as a function of the BH fraction. Uses an exponential fit to connect minimum and maximum values of b.
        
        # Integration parameters.
        self.tend = 13.8e3 # [Myrs] Final time instance for inegration. Here taken to be a Hubble time.
        self.dtout = 2 # [Myrs] Time step for integration. Default to a few Myrs. It is not required, can be set equal to None.
        self.Mst_min = 100 # [Msun] Stop criterion for stars. Below this value, the integration stops.
        self.Mbh_min = 500 # [Msun] Stop criterion for BHs. Below this value the integration stops. The integrator stops only if both the stellar option is selected and the specified conditions are met.
        self.integration_method = "RK45" # Integration method. Default to a Runge-Kutta.
        self.vectorize = False # Condition for solve_ivp. Solves faster coupled differential equations when the derivatives are arrays.
        self.dense_output = True # Condition for solve_ivp to give continuous solution. Facilitates interpolation. Can be used instead of specifying t_eval.
        self.rtol, self.atol = 1e-6, 1e-7 # Relative and absolute tolerance for integration.
        
        # Output.
        self.output = False # A Boolean parameter to save the results of integration along with a few important quantities.
        self.outfile = "cluster.txt" # File to save the results, if needed.

        # Conditions.
        self.ssp = True # Condition to use the SSP tools to extract the BHMF at any moment. Default option uses such tools. 
        self.sev = True # Condition to consider effects of stellar evolution. Affects total mass and expansion of the cluster. Default option considers stellar evolution.
        self.kick = True # Condition to include natal kicks. Affects the BH population obtained. Default option considers kicks.
        self.tidal = True # Condition to activate tides. Default option considers the effect of tides.
        self.escapers = False # Condition for escapers to carry negative energy as they evaporate (only) from the cluster due to the tidal field. Affects mainly the expansion of sparse clusters at small galactocentric distances. Default option is deactivated.
        self.varying_kin = False # Condition to include a variation of the kinetic term with the number of stars. It is combined with 'escapers'. Default option is deactivated.
        self.two_relaxations = True # Condition to have two relaxation time scales for the two components. Differentiates between ejections and evaporation. Default option is activated.
        self.mass_segregation = False # Condition for mass segregation. If activated, parameter Mval evolves from Mval0 up until Mvalf within one relaxation. Default option is deactivated.
        self.Virial_evolution = False # Condition to consider different evolution for the Virial radius from the half-mass radius within one relaxation. Default option is deactivated.
        self.psi_exponent_run = False # Condition to have a running exponent in parameter ψ based on the BH fraction. Default option is deactivated.
        self.finite_escape_time = False # Condition to implement escape from Lagrange points for stars which evaporate. Introduces an additional dependence on the number of particles on the tidal field, prolongs the lifetime of clusters. Default option is deactivated.
        self.running_bh_ejection_rate_1 = True # Condition for decreasing the ejection rate of BHs due to varying E / M φ0. Default option considers the effect of decreasing fbh only. Currently activated.
        self.running_bh_ejection_rate_2 = True # Condition for decreasing the ejection rate of BHs due to equipartition. Uses a smooth function for decreasing the BH ejection efficiency. Currently activated.
        self.running_stellar_ejection_rate = False # Condition for changing the stellar ejection rate compared to the tidal field. Currently deactivated.
        self.rotation = False # Condition to describe rotating clusters. Currently it is inserted as a simple extension, more study is required. It is deactivated.
        self.dark_clusters = False # Condition for describing dark clusters (stellar population depletes first). When used, the approximations in ψ are not used. It is regarded as future extension and needs to be augmented. Currently it is deactivated
        
        # Stellar evolution model. Used for selecting a model for ν.
        self.sev_model = 'constant' # Default option is constant mass loss rate.
        
        # Motion of cluster.
        self.Vc = 220. # [km/s] Circular velocity of cluster, for instance inside a singular isothermal sphere (SIS). 
        
        # Galactic model. Used for specifying the tidal radius.
        self.galactic_model, self.rt_index = 'SIS', 2 # Default to 'SIS'. Affects the tidal radius. 
        
        # Tidal model. Used for specifying the evaporation rate ξ.
        self.tidal_model = 'Power_Law' # Default to 'Power_Law'. Affects tidal mass loss and thus the evolution of the half-mass radius.
        
        # Cluster model. Used for specifying the potential of the cluster.
        self.cluster_model = 'Point_mass' # Default to 'Point_mass'. Currently it affects only the energy contribution of evaporating stars, if activated.
        
        # Model for the BH ejection rate β.
        self.beta_model = 'exponential' # Default option to the exponential model. Affects the dependence of the BH ejection rate on Spitzer's parameter and thus the extrapolation to small BH populations or large metallicity clusters.
        
        # Balance model. It can be used for a smooth connection of the unbalanced phace with post core collapse.
        self.balance_model = 'error_function' # Default option to error function. Uses a function in order to smoothly connect the unbalanced phase to post core collapse. Serves as a future implementation to make the differential equations continuous.
        
        # Pass in lists for a_slopes, m_breaks and nbins. Any number of IMF intervals, based on size of lists. m_breaks must be size N+1
       
        # Slopes of mass function.
        self.a_slopes = [-1.3, -2.3, -2.3]
        self.a_kroupa = self.a_slopes.copy() # In case the user inserts a different set of slopes (and mass breaks), the stellar mass loss rate ν is different.

        # Mass function break masses.
        self.m_breaks = [0.08, 0.5, 1., 150.]
        self.m_kroupa = self.m_breaks.copy()

        # Number of bins per interval of the mass function.
        self.nbins = [5, 5, 20]

        # All other arguments to be passed to InitialBHPopulation, including IFMR and natal kick options. See SSPtools documentation for details.
        self.ibh_kwargs = dict()
        
        # Check input parameters, then proceed with the computations.
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        self.nu_factor = 0 # Factor that corrects solution for Mst, mst for the case of a different IMF that has similar upper IMF with Kroupa. Default to 0.

        # Check that this is not exactly a Kroupa IMF
        if (not (numpy.array_equal(self.a_slopes, self.a_kroupa)
                 and numpy.array_equal(self.m_breaks, self.m_kroupa))):

            # Check if IMF high mass slope/breaks match Kroupa above 1Msun
            if ((self.a_slopes[-1] == self.a_kroupa[-1]) and numpy.array_equal(self.m_breaks[-2:], self.m_kroupa[-2:])):

                self.nu_factor = 1 - self._sev_factor(self.a_kroupa, self.a_slopes, self.m_kroupa, self.m_breaks) # Approximate expression for auxiliary factor. It is inserted in the differential equations for Mst, mst

            # If the upper part of the IMF changes, tsev needs to change as well.
            else:
                import warnings
                warnings.warn('IMF slope >1Msun is different than a Kroupa IMF, new stellar evolution rate cannot be automatically computed. '
                              f'Using given {self.nu=}.')

        # Define models for stellar evolution. Coefficients can capture different dependences on metallicity, or an explicit dependence can be inserted.
        self.sev_dict={
            'constant': lambda Z, t: self.nu,
            'exponential': lambda Z, t: self.nu * exp (- self.c1 * ((t - self.tsev) / self.tsev) ** self.c2),
            'power_law': lambda Z, t: self.nu / (1 + self.c1 * ((t - self.tsev) / self.tsev) ** self.c2),
            'logarithmic': lambda Z, t: self.nu * (1 + self.c1 * log(t / self.tsev) - self.c2 * log(t / self.tsev) ** 2)
        }
        
        # Galactic properties.
        self.Mg = 1.023 ** 2 * self.Vc ** 2 * self.rg * 1e3 / self.G # [Msun] Mass of the galaxy inside the orbit of the cluster. It is derived from orbital properties.
        self.L = 1.023 * self.Vc * self.rg * 1e3 # [pc^2 / Myrs] Orbital angular momentum.
        
        # Define the tidal models dictionary.
        self.tidal_models = { # Both the half-mass and the tidal radius should be in [pc]. All the rest parameters are fixed. In a different scenario, they should be treated as variables.
            'Power_Law': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n, 
            'Exponential': lambda rh, rt: 3 * self.zeta / 5 * exp(self.c * rh / rt)
        }
        
        # Galactic model dictionary.
        self.galactic_model_dict = { # Dictionary for spherically symmetric galactic potentials. A model can be selected from the dictionary. The index is used for the tidal radius only and it is dimensionless. The potential has units are [pc^2 / Myr^2]. Radius should be in [kpc], mass in [Msun].
            'SIS': {'rt_index':2 , 'phi':lambda r, Mg: self.Vc ** 2 * log(r / self.rp)}, # Singular isothermal sphere. It is the default.
            'Point_mass': {'rt_index':3, 'phi': lambda r, Mg: - 1e-3 * self.G * Mg / r }, # Point mass galaxy.
            'Hernquist': {'rt_index':(3 * self.rg + self.rp) / (self.rg + self.rp), 'phi': lambda r, Mg: - 1e-3 * self.G * Mg / (r + self.rp) },  # Hernquist model.
            'Plummer': {'rt_index':3 * self.rg ** 2 / (self.rg ** 2 + self.rp ** 2), 'phi':lambda r, Mg: - 1e-3 * self.G * Mg / sqrt(r ** 2 + self.rp ** 2)},  # Plummer model.
            'Jaffe': {'rt_index':(3 * self.rg + 2 * self.rp) / (self.rg + self.rp), 'phi':lambda r, Mg: - 1e-3 * self.G * Mg / self.rp * log(1 + self.rp / r)},   # Jaffe model.
            'NFW': {'rt_index':1 + (2 * log(1 + self.rg / self.rp) - 2 * self.rg / (self.rg + self.rp) - (self.rg / (self.rg + self.rp)) ** 2 ) / (log(1 + self.rg / self.rp) - self.rg / (self.rg + self.rp)), 'phi':lambda r, Mg:- 1e-3 * self.G * Mg / r * log(1 + r / self.rp)}, # Navarro-Frank-White model.
            'Isochrone': {'rt_index':1 + 1 / (self.rp + sqrt(self.rg ** 2 + self.rp ** 2)) / sqrt(self.rg ** 2 + self.rp ** 2) * (2 * self.rg ** 2 - self.rp ** 2 * (self.rp + sqrt(self.rg ** 2 + self.rp ** 2)) / sqrt(self.rp ** 2 + self.rg ** 2)), 'phi':lambda r, Mg:- 1e-3 * self.G * Mg / (self.rp + sqrt(r ** 2 + self.rp ** 2))}
        } 
        
        # Cluster model dictionary.
        self.cluster_model_dict = { # Dictionary for spherically symmetric cluster potentials. Units are [pc^2 / Myr^2]. Radius should be in [pc], mass in [Msun].
            'SIS': lambda r, M: self.G * M / r * log(r / self.rpc),
            'Point_mass': lambda r, M: - self.G * M / r,
            'Plummer': lambda r, M: - self.G * M / sqrt(r ** 2 + self.rpc ** 2),
            'Hernquist': lambda r, M: - self.G * M / (r + self.rpc),
            'Jaffe': lambda r, M: - self.G * M / self.rpc * log(1 + self.rpc / r),
            'NFW': lambda r, M: - self.G * M / r * log(1 + r / self.rpc),
            'Isochrone': lambda r, M: - self.G * M / (self.rpc + sqrt(r ** 2 + self.rpc ** 2))
        }
        
        # Dictionary that introduces a scaling on the BH ejection rate with respect to Spitzer's paramater. Can be used as a proxy for different parametrizations as well.
        self.beta_dict = {
           'exponential': lambda S: 1 - exp(- (S / self.S0) ** self.gamma2),
           'logistic1': lambda S: (S / (S + self.S0)) ** self.gamma2,
           'logistic2': lambda S: (S ** self.gamma2 / (S ** self.gamma2 + self.S0 ** self.gamma2)),
           'error_function': lambda S:  erf((S + self.S_crit) / self.S0),
           'hyperbolic': lambda S: tanh((S + self.S_crit) / self.S0)
        }
        
        # Dictionary for balancing functions. It is currently on trial stage and is not used. Functions must be 0 at t=0 and for initial times, and 1 at tcc<=t.
        self.balance_dict = {
            'error_function':lambda t:  0.5 * (1 + erf(self.gamma4 * self.gamma3 * (t - self.gamma4 * self.tcc / self.gamma3 ) / self.tcc)) ,
            'hyperbolic': lambda t: 0.5 * (1 + tanh(self.gamma3 * self.gamma4 * (t - self.gamma4 * self.tcc / self.gamma3) / self.tcc)),
            'exponential': lambda t: 1 - exp(- self.gamma3 * (t / self.tcc) ** self.gamma4)
        }
        
        # The user can either specify the initial average mass beforehand, or it is extracted from the IMF.
        # If m0 is specified in the beginning, the IMF must be well defined so that it matches.
        if self.m0 is None:
           self.m0 = self._initial_average_mass(self.a_slopes, self.m_breaks) # [Msun] Average mass obtained for this particular IMF. 
        
        self.FeH = log10(self.Z / self.Zsolar) # Metallicity in solar units.
        
        self.M0 = self.m0 * N # [Msun] Total mass of stars (cluster) initially. This is used for computations, even if the BH population is present in clusterBH at t=0.
        self.rh0 = (3 * self.M0 / (8 * pi * rhoh)) ** (1./3) # [pc] Initial half-mass radius.
        self.vesc0 = self._vesc(self.M0, self.rh0) # [km/s] Initial escape velocity.
        
        # Check the BH ejection model.
        if self.sev_model not in self.sev_dict:
            raise ValueError(f"Invalid model for stellar evolution: {self.sev_model}.")
        
        # Check the BH ejection model.
        if self.beta_model not in self.beta_dict:
            raise ValueError(f"Invalid model for BH ejections: {self.beta_model}.")
        
        # Check the balance model. Future implementation.
        if self.balance_model not in self.balance_dict:
            raise ValueError(f"Invalid model for balancing: {self.balance_model}.")
        
        # Check if there is a step for integration. If so, at each step the results are saved. Otherwise a dense output is used.
        if self.dtout is None:
            self.t_eval = None
        else:
            self.t_eval = numpy.arange(0, self.tend, self.dtout) # Area for integration.
        
        self.mst_sev, self.t_mst = self._mst_sev() # [Msun] Solution for average stellar mass if stellar evolution is considered solely. Time is kept for interpolation.
        
        # Check whether SSP are used for the BHMF.
        if self.ssp:
            import ssptools # Import the package. Must be installed first.
            
            self.ibh_kwargs.setdefault('kick_vdisp', self.sigmans) # Default arguments for ssp.
            
            # Implement kicks, if activated, for this IMF, number of stars, with such metallicity, central escape velocity and BHMF conditions.
            self.ibh = ssptools.InitialBHPopulation.from_powerlaw(self.m_breaks, self.a_slopes, self.nbins, self.FeH, N0=N, vesc=self.vesc0, natal_kicks=self.kick, **self.ibh_kwargs)
            self.Mbh0 = self.ibh.Mtot  # [Msun] Expected initial mass of BHs.
            self.f0 = self.Mbh0 / self.M0 # Initial fraction of BHs. Should be close to 0.06 for poor-metal clusters. 
            
            self.Nbh0 = numpy.round(self.ibh.Ntot).astype(int) # Initial number of BHs. Round the number for consistency.
            self.mbh0 = self.Mbh0 / self.Nbh0 # [Msun] Initial average BH mass.
            self.mlo = self.ibh.m.min() # [Msun] Minimum BH mass in the BHMF.
            self.mup = self.ibh.m.max() # [Msun] Maximum BH mass in the BHMF.
            self.Mst_lost = self.ibh.Ms_lost # [Msun] Mass of stars lost in order to form BHs.
            self.t_bhcreation = self.ibh.age # [Myrs] Time needed to form these astrophysical mass BHs.
            self.mst_sev_BH = numpy.interp(self.t_bhcreation, self.t_mst, self.mst_sev) # [Msun] Average stellar mass when all BHs are created. Effect of tides is neglected for simplicity.
            
        # Should the user wishes to exclude the ssp tools and use another approach, they can first define f0 and then compute Mbh0. A simple approach would be the same value regardless of kicks.
        else:
            mmax_ = numpy.logspace(log10(self.mlo), log10(self.mup), self.N_points) # List of possible values for the maximum BH mass at a given time instance.
            self.b_BH = (self.alpha_BH + 2) / 3 # Auxiliary exponent.
            self.qul = self.mup / self.mlo # Ratio of upper and lower mass in the BHMF.
            self.qul3, self.mlo3b, self.mup3b = self.qul ** 3, self.mlo ** (3 * self.b_BH), self.mup ** (3 * self.b_BH) # Constants that are used multiple times. They are defined to avoid subsequent recomputations.
            if self.alpha_BH != -2: self.b_BH3 = 1./(3 * self.b_BH) # Exponent that participates in computations for alpha_BH different than -2, for no kicks.
            
            self.Mbh0 = self.f0 * self.M0 # [Msun] Prediction for the initial BH mass if SSP are not used and kicks are deactivated.
            
            if self.kick: # Depending on the condition for kicks, a different retention fraction is used.
               
               # Checks if the user has actually inserted a bad exponent for the BHMF that cannot be used.
               k = (self.alpha_BH + 5) / -3  # Solve for k in equation: self.alpha_BH = -5 - 3k. These are values that give negative integers in the third input of the hypergeometric functions used.
    
               if k.is_integer() and k >= 0:  # Checks whether k is a non-negative integer.
                  raise ValueError(f"Exponent {self.alpha_BH} cannot be used with kicks and power-law BHMF since the hypergeometric functions used are ill defined. Choose other conditions.")

               self.mb = (9 * pi / 2) ** (1./6) * self.sigmans * self.mns / self.vesc0 # [Msun] Correct the lowest mass unaffected by kicks. All masses below this threshold are affected by kicks.
               self.qub, self.qlb = self.mup / self.mb, self.mlo / self.mb # Mass ratios.
               self.qub3, self.qlb3, self.qul3b = self.qub ** 3, self.qlb ** 3, self.qul ** (3 * self.b_BH) # Ratios that appear multiple times in calculations.
               self.h2 = hyp2f1(1, self.b_BH, self.b_BH + 1, -self.qlb3) # Hypergeometric function. Appears several times.
               self.mloh2 = self.mlo3b * (1 - self.h2) # Constant prefactor which appears several times.
               
               # Function that returns the result of integrating the power-law mass function. It is normalized to the prefactor of the mass function. Valid only for power-law BHMF.
               def integr(mm, qmb):

                   if self.alpha_BH == -2: # This excludes only the alpha_BH = -2, but in principle other choices should be excluded, those that are problematic for hypergeometric functions (negative integers for b + 1). 
                      return log((1 + qmb ** 3) / (1 + self.qlb3))  # Integral of power-law mass function without the constant prefactor.
                   
                   else:
                       
                      h1 = hyp2f1(1, self.b_BH, self.b_BH + 1, -qmb ** 3) # Hypergeometric function used for the solution.
                      return mm ** (3 * self.b_BH) * (1 - h1) - self.mloh2 # BH mass extracted by integrating the mass function, neglecting the constant prefactor.

               self.qmb = mmax_ / self.mb # Auxiliary mass rations that can be used in the solution.

               if self.alpha_BH != -2: # It is important so that the hypergeometric function is well defined.
                   h1 = hyp2f1(1, self.b_BH, self.b_BH + 1, -self.qub3) # Hypergeometric function, part of the solution.
                   self.fretm = 1 - (self.qul3b * h1 - self.h2) / (self.qul3b - 1) # Retention fraction.
               
               else:
                   
                   self.fretm = log( (self.qub3 + 1) / (self.qlb3 + 1) ) / log(self.qul3) # Retention fraction.
               
               self.Mbh0 *= self.fretm # [Msun] Correct the Initial BH mass since the retention fraction is now lowered.
               
            # Compute function values to be used in integrals.
            num_values = numpy.array(mmax_ ** (self.alpha_BH + 1) / ((self.mb / mmax_) ** 3 + 1))
            den_values = numpy.array(mmax_ ** self.alpha_BH / ((self.mb / mmax_) ** 3 + 1))

            # Precompute integrals for average mass.
            self.num_integral = cumulative_trapezoid(num_values, mmax_, initial=0) # [Msun] Total mass in the BHMF. 
            self.den_integral = cumulative_trapezoid(den_values, mmax_, initial=0) # Total number of BHs in the BHMF.
            self.mmax_ = mmax_ # Array will be recalled.
            self.Mbh_ = self.Mbh0 * self.num_integral / self.num_integral[-1] #  [Msun] BH mass points that are generated from the possible max BH mass points.
            self.Mbh_[self.Mbh_ < self.mlo] = self.mlo # Ensure that no value is below the lowest BH mass.
            
            # Further properties.
            self.mbh0 = self._mbh(self.Mbh0) # [Msun] Initial average BH mass.
            self.Nbh0 = numpy.round(self.Mbh0 / self.mbh0).astype(int) # Initial number of BHs.
            self.mst_sev_BH = numpy.interp(self.t_bhcreation, self.t_mst, self.mst_sev) # [Msun] Average stellar mass when all BHs are created. Effect of tides is neglected for simplicity.
            self.Mst_lost = (self.m0 - self.mst_sev_BH) * N # [Msun] Approximate value for the stellar mass lost to create all BHs.
        
        # Initial relaxation is extracted with respect to the BH population that will be obtained. It is an approximation such that large metallicity clusters have larger relaxation, as expected. This approach inserts an indirect metallicity dependence.
        self.psi0 = self._psi(self.Mbh0 / (self.M0 + self.Mbh0 - self.Mst_lost), self.M0 + self.Mbh0 - self.Mst_lost, self.mbh0, self.mst_sev_BH) # Use as time instance the moment when all BHs are formed. In this regime, tides are not so important so all the mass-loss comes from stellar evolution. 
        self.trh0 = 0.138 / (self.m0 * self.psi0 * log(self.gamma * self.N)) * sqrt(self.M0 * self.rh0 ** 3 / self.G ) * (1 - 2 * self.omega0 ** 2 * self.rh0 ** 3 / (self.G * self.M0)) ** (3 / 2) # [Myrs] We use m0 since at t=0 the mass distribution is the same everywhere.
        self.tcc = self.ntrh * self.trh0 # [Myrs] Core collapse. Effectively it depends on metallicity, size and mass of the cluster.
        
        # Check if the galactic model, cluster model, tidal model have changed. Based on the available choices in the dictionaries, change the values accordingly.
        if self.tidal: # Valid check only if we have tides.
            if self.galactic_model in self.galactic_model_dict:
                self.rt_index = self.galactic_model_dict[self.galactic_model]['rt_index']
            else:
                raise ValueError(f"Invalid galactic model: {self.galactic_model}.") 
                
            # Currently the cluster model is needed only if tides are activated and the evaporation of stars changes the energy budget.
            if self.cluster_model not in self.cluster_model_dict:
                raise ValueError(f"Invalid cluster model: {self.cluster_model}.") 
        
            if self.tidal_model not in self.tidal_models:
                raise ValueError(f"Invalid tidal model: {self.tidal_model}.")
        
        # Specify the functions that shall be used. They are available in the dictionaries.
        self.nu_function = self.sev_dict[self.sev_model]
        self.xi_function = self.tidal_models[self.tidal_model]
        self.beta_function = self.beta_dict[self.beta_model]
        self.cluster_model_function = self.cluster_model_dict[self.cluster_model]
       
        self._evolve(N, rhoh) # Runs the integrator, generates the results.

    def _sev_factor(self, a_slopes1, a_slopes2, m_breaks1, m_breaks2):
            
        def integrate_IMF(m_low, m_high, p):
            if p == -2:  # Handle special case where p = -2 to avoid division by zero
                return log(m_high / m_low)
            else:
                return (m_high ** (p + 2) - m_low ** (p + 2)) / (p + 2)

        def norm_constants(a_slopes, m_breaks):
            c_values = [1.0]  # c1 = 1.0 (initial normalization)
            norm_constants = []
        
            for i in range(1, len(a_slopes)):
                c_values.append(m_breaks[i] ** (a_slopes[i - 1] -a_slopes[i]))
        
            for i in range(len(c_values)):
                norm_constants.append(numpy.prod(c_values[:i+1]))
        
            return norm_constants
    
        # Compute normalization constants for both sets of slopes
        norm_consts1 = norm_constants(a_slopes1, m_breaks1)
        norm_consts2 = norm_constants(a_slopes2, m_breaks2)
        
        # Compute the integral for both IMF sets.
        integral1, integral2 = 0, 0
        for i in range(len(a_slopes1)):
            integral1 += norm_consts1[i] * integrate_IMF(m_breaks1[i], m_breaks1[i + 1], a_slopes1[i])

        for i in range(len(a_slopes2)):
            integral2 += norm_consts2[i] * integrate_IMF(m_breaks2[i], m_breaks2[i + 1], a_slopes2[i])
        
        # Fraction of mass in the upper limit of the Kroupa.
        r = norm_consts2[-1] * integrate_IMF(m_breaks2[-2], m_breaks2[-1], a_slopes2[-1])
        r /= norm_consts1[-1] * integrate_IMF(m_breaks1[-2], m_breaks1[-1], a_slopes1[-1])
       
        # Return the ratio of the two integrals.
        return r * integral1 / integral2

    
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
       - It assumes the IMF is continuous across the mass breakpoints.
       - Special handling is included for slopes to avoid division by zero.
       """
       
       def integrate_IMF(m_low, m_high, p):
        
           if p == -1:  # Special case where p = -1 to avoid division by zero, should the user select such value.
              return log(m_high / m_low) 
           else:
              return (m_high ** (p + 1) - m_low ** (p + 1)) / (p + 1)
       
       # Normalization constants to ensure continuity of the IMF.
       c_values = [1.0]  # c1 = 1.0
       normalization_constants = []
       
       # Calculate the rest of the c values based on slopes and mass breakpoints. This is for continuity.
       for i in range(1, len(a_slopes)):
           c_values.append(m_breaks[i] ** (a_slopes[i - 1] - a_slopes[i])) # Previous exponent minus current one, to compute the current prefactor.

       # Compute the cumulative products to form the normalization_constants array.
       for i in range(len(c_values) - 1):
         normalization_constants.append(c_values[i] * c_values[i + 1]) # For every prefactor, we need to take the product with the previous one. We have 2 breakpoints in 1 interval.
       
      # The prefactors are dimensionless here, if a normalization to 1Msun is assumed in the power-laws.
       normalization_constants.insert(0, c_values[0]) # First entry is 1. It does not matter because the normalized prefactors are essentially conputed.
       
       # Set initial values to 0.
       stellar_mass = 0 
       stellar_number = 0
       
       # For every pair, compute the number of massess and stars.
       for i, a in enumerate(a_slopes):
          m_low = m_breaks[i]
          m_high = m_breaks[i + 1]
          c_i = normalization_constants[i]
          
          # Compute numerator.
          stellar_mass += c_i * integrate_IMF(m_low, m_high, a + 1)
         
          # Compute denominator.
          stellar_number += c_i * integrate_IMF(m_low, m_high, a)
    
       # Calculate the average mass. A well defined IMF should have a nonzero denominator.
       return stellar_mass / stellar_number
    
    # Function to determine by how much the BHMF changes, given a particular mass loss.
    def _deplete_BHMF(self, M_eject, M_BH, N_BH):
        """
        Deplete the BHMF by ejecting mass starting from the heaviest bins. This is subject to change once dark clusters are considered. For the time being, it suffices.
    
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
 
        # Bins are emptied for as long as the mass ejected is nonzero.
        while M_eject != 0: # [Msun]
            j -= 1

            # Stop ejecting if trying to eject more mass than there is in BHs.
            if j < 0:
                break
             
            # Remove entirety of this mass bin. Adjust the ejected mass.
            if M_BH[j] < M_eject: # [Msun]
                M_eject -= M_BH[j] # [Msun]
                M_BH[j] = 0 # [Msun]
                N_BH[j] = 0 # Dimensionless
                continue

            # Remove required fraction of the last affected bin.
            else:
                m_BH_j = M_BH[j] / N_BH[j] # [Msun] The average BH mass in a given bin is assumed to be constant.
                M_BH[j] -= numpy.asarray(M_eject).item()  # [Msun] Convert to scalar explicitly, allows for multiple values to be computed.
                N_BH[j] -= numpy.asarray(M_eject / m_BH_j).item() # Number of remaining BHs.
                break

        return M_BH, N_BH
    
    # Calculates the upper mass in the BHMF at any given moment.
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
        
        # Check the routine that is used for the BHMF.
        if self.ssp:
            
            M_eject = self.Mbh0 - Mbh # [Msun] Mass that has been ejected.
            M_BH, N_BH = self._deplete_BHMF(M_eject, self.ibh.M, self.ibh.N) # [Msun, dimensionless]. BHMF at a given time instance.
            valid_bins = N_BH > 0 # Create a mask to see which bins are not empty. These have BHs.
            # Find the maximum individual BH mass in the valid bins.

            if valid_bins.any() > 0: # Ensure valid_bins is not empty
               mmax = numpy.max(self.ibh.m[valid_bins]) # [Msun] Consider only non-zero bins.
            else:
               mmax = 0 # All BHs have been ejected.
        
        else:
           
            # Arrays Mbh_ and mmax_ are constructed accordingly for all cases of alpha_BH, apart from those that are ill defined. No need to check.
            mmax = numpy.interp(Mbh, self.Mbh_, self.mmax_)
            
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
        
        # Check which BHMF is used.
        if self.ssp:
           # Determine amount of BH mass (total) to eject.
           M_eject = self.Mbh0 - Mbh # [Msun]
            
           # Deplete the initial BH MF based on this M_eject.
           M_BH, N_BH = self._deplete_BHMF(M_eject, self.ibh.M, self.ibh.N) # [Msun, dimensionless]
         
           return M_BH.sum() / N_BH.sum() # [Msun]
        
        else: 
            
            mmax = self._find_mmax(Mbh) # [Msun] Upper BH mass in the BHMF. Assumes a simple power-law function for the MF with slope a_BH.
            mmax_ = self.mmax_ # Call the array for upper masses. It is used multiple times.
            
            # Interpolate for each mmax. Find the total mass and number of BHs. The integrals are precomputed.
            numerator = numpy.interp(mmax, mmax_, self.num_integral)
            denominator = numpy.interp(mmax, mmax_, self.den_integral)
            # Perform the integration. Assume a power-law BHMF, corrected for kicks. If kicks are deactivated, mb=0 by default. For a different model instead of power-law, the integrands change but also the routine for kicks in _find_mmax.
         
            return numerator / numpy.maximum(denominator, 1e-99) # [Msun]
            return numerator / numpy.maximum(denominator, 1e-99) # [Msun]
       
    # Tidal radius. Applies for any orbit on spherically symmetric potentials. If axisymmetric are used, a different prescription is needed.
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

    # Average mass of stars, due to stellar evolution.    
    def _mst_sev(self):
        """
        Solves the differential equations for mst and Mst over the time interval,
        considering mst = m0, Mst=M0 for t < self.tsev. 

        Returns:
        -------
        Array: The average stellar mass [Msun] as it would evolve only through stellar evolution.
        Array: The time steps of the solution [Myrs] to be used for interpolation.
        """
        
        def evolution(t, y):
            mst, Mst = y
            if not self.sev:
               return [0, 0]
            sev_value = numpy.maximum(self.sev_dict[self.sev_model](self.Z, t), 0)
            factor = sev_value / (t + 1e-99) * numpy.heaviside(t - self.tsev, 1)
            return [-factor * (mst - mst * self.M0 / Mst * self.nu_factor),
                -factor * (Mst - self.M0 * self.nu_factor)]

        sol = solve_ivp(evolution, [0, self.tend], [self.m0, self.M0], 
                    method=self.integration_method, t_eval=self.t_eval, 
                    rtol=self.rtol, atol=self.atol, vectorized=self.vectorize, 
                    dense_output=self.dense_output)
       
        # Return the value of mst and time as arrays. The latter will be used for interpolation.
        return sol.y[0], sol.t
        
    # Friction term ψ in the relaxation. It is characteristic of mass spectrum within rh, here due to BHs. We neglect O(2) corrections in the approximate form.
    def _psi(self, fbh, M, mbh, mst): 
        """
        Calculate ψ for the cluster based on various parameters.

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
           The calculated ψ value, which depends on the BH fraction, cluster mass and average masses.

        Notes:
        ------
        - The function assumed that the input values are well defined (for Mbh < mbh, both should be 0, same for Mst < mst).
        - The number of particles `Np` is calculated using the `_N` function, which depends on the cluster's mass, 
          the BH fraction, and the average masses of the populations.
        - The average stellar mass `mav` is derived by dividing the total cluster mass by the number of particles.
        - The exponent `gamma` is either a constant value `b` or it evolves based on the BH fraction (if `psi_exponent_run` is True),
          starting from equal velocity dispersion for different particles and evolving towards equipartition as time passes.
        - The expression for `psi` includes a combination of parameters like `a0`, `a11`, `a12`, `b1`, `b2`, and others, 
          which relate the properties within the cluster's half-mass radius (`rh`) to the global properties of the cluster.
        - For dark clusters (`dark_clusters` is True), a more complex expression for `psi` is used, 
          incorporating contributions from both the stellar and BH populations.
        - From t=0 until tcc, a better expression may be needed, one that accounts for initial stellar mass spetrum and how the two populations evolve up until tcc. This will allow to bypass the insertion of psi0 in trh0, before tcc is computed initially.
        """
        
        # Number of particles.
        Np = M * ( fbh / mbh + (1 - fbh) / mst)
       
        # Average mass and stellar mass.
        mav = M / Np # [Msun]
       
        # Exponent.
        gamma = self.b if not self.psi_exponent_run else self.b_min + (self.b_max - self.b_min) * exp (- self.gamma_exp * fbh) # No BHs, starts with equal velocity dispersion. As time flows by it approaches equipartition. Maximum value is 2.26.
        
        # Result.
        if not self.dark_clusters:
            # Approximate form is the default option. Parameters a11, a12, b1 and b2 relate the properties within rh to the global properties.
            psi = self.a0  + self.a11 * self.a12 ** (gamma - 1) * fbh ** self.b1 * (mbh / mav) ** ((gamma - 1) * self.b2)  
        else:
            # Total expression. Can be used for instance if the stellar population depletes first.
            psi = self.a0 * (mst / mav) ** (self.b0 * (gamma - 1)) * (1 - self.a11 * fbh ** self.b1) + self.a11 * self.a12 ** (gamma - 1) * fbh ** self.b1 * (mbh / mav) ** (self.b2 * (gamma - 1)) # Complete expression if we include the contribution from stars as well.
           
        return psi
    
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
        - The function assumed that the input values are well defined (for Mbh < mbh, both should be 0, same for Mst < mst, while cases where M < 0 or rh < 0 should not be used).
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass within `rh` is computed using a power-law fit based on the total mass and the number of particles.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, particle number `Np`
          and the gravitational constant `G`. The timescale is further modified by the cluster's `psi` value, which depends on the BH fraction.
        - If the cluster undergoes rigid rotation (`self.rotation` is True), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model. The expression by King is used.
        """
        
        # Number of particles.
        Np = M * ( fbh / mbh + (1 - fbh) / mst) 
        
        # Average mass within rh. We use a power-law fit.
        mav =  self.a3 * (M / Np) ** self.b3  # [Msun]
        
        # Mass spectrum within rh.
        psi = self._psi(fbh, M, mbh, mst)
        
        # Relaxation.
        trh = 0.138 * sqrt(M * rh ** 3 / self.G) / (mav * psi * log(self.gamma * Np)) # [Myrs]. Solution stops when the number of particles is low so the logarithm is well defined.
        
        if self.rotation: # Effect of rigid rotation.
            trh *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** (3 / 2) # [Myrs] Assume constant rotation for now.
        
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
        - The function assumed that the input values are well defined (for Mbh < mbh, both should be 0, same for Mst < mst, while cases where M < 0 or rh < 0 should not be used).
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass within the half-mass is computed from the total mass and number of particles for the whole cluster.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, particle number `Np`
          and the gravitational constant `G`. 
        - If the cluster undergoes rigid rotation (`self.rotation` is True), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model assuming constant rotation.
        """
        
        # Number of particles.
        Np = M * ( fbh / mbh + (1 - fbh) / mst)
        
        # Relaxation for evaporation.
        trhstar = 0.138 * sqrt(M * rh ** 3 / self.G) / (mst *  log(self.gamma * Np)) # [Myrs]
      
        if self.rotation: # Effect of rigid rotation.
            trhstar *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** (3 / 2) # [Myrs] Assume constant rotation for now.
          
        return trhstar
          
    # Crossing time within the half-mass radius in Myrs.
    def _tcr(self, M, rh, k=1): 
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
        - The function assumed that the input values are well defined ( M < 0 or rh < 0 should not be used).
        - The value of `k` is adjusted for different cases, for instance at the tidal radius.
        - The crossing timescale is calculated using the formula: `tcr = k * sqrt(rh^3 / (G * M))`, where `rh` is the half-mass radius and `M` is the total mass of the cluster.
          It is derived from `tcr = 1 / sqrt(G rhoh)` or `tcr = rh / sqrt(-2E)` according to the Virial theorem. Both give the same scaling.
        """
        
        tcr = k * sqrt(rh ** 3 / (self.G * M)) # [Myrs] With rotation, the numerial values of rh, M change and this impacts crossing time indirectly.
      
        return tcr
    
    # Central escape velocity.
    def _vesc(self, M, rh):
        """
        Calculate the central escape velocity (`vesc`) for a cluster, based on its total mass and half-mass radius.

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
          `rhoh = 3 * M / (8 * pi * rh ** 3)`. The density is in units of [Msun / pc^3].
        - The escape velocity `vesc` is then calculated using the relation: 
          `vesc = 50 * (M / 1e5) ** (1/3) * (rhoh / 1e5) ** (1/6)`, where the units for `vesc` are in [km/s]. 
          This formula expresses the central escape velocity based on the mass and the density of the cluster.
        - The escape velocity is further augmented by multiplying it by the factor `fc`, which adjusts the value according to the specific King model used for the cluster.
        - Time dependence if `fc` is neglected, however it should increase over time. This is introduced, currently as a trial, in a different function.
        """
        
        # Density.
        rhoh = 3 * M / (8 * pi * rh ** 3) # [Msun / pc^3]
        
        # Escape velocity.
        vesc = 50 * (M / 1e5) ** (1./3) * (rhoh / 1e5) ** (1./6) # [km/s] Central escape velocity as a function of mass and density. If it is needed in [pc / Myrs], multiply with 1.023.
        vesc *= self.fc # Augment the value for different King models. This in principle evolves with time but the constant value is an approximation.
       
        return vesc

    # Evolution of prefactor fc over time. A trial dependence on the macroscopic properties of the cluster is assumed.
    def _fc(self, M, rh):
        
        """
        Estimate the change in parameter fc, and effectively on King's parameter W0, and its impact on the central escape velocity.
        
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
        
        fc = (self.M0 / M) ** self.d1 * (rh / self.rh0) ** self.d2 # Such parameter should increase because the central region becomes denser over time. Will be studied further in the future.
        
        return fc
    
    # Construct the differential equations to be solved. 
    def _odes(self, t, y):
        """
        Compute the time derivatives of the system's state variables, which describe the evolution of a star cluster 
        under various physical processes including stellar evolution, tidal effects, ejections, mass segregation, and core collapse.

        Parameters:
        -----------
        t : float
           Time elapsed since the beginning of the cluster's evolution, in [Myrs].
    
        y : array
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
           - `Mbh_dot`: BH mass loss rate [Msun/Myr].
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
         - Stellar mass loss due to stellar evolution is modeled using mass-loss rate (`nu`). 
         - Mass segregation is evolved if the option is selected.
         - Virial radius evolves differently in the unbalanced phase if the option is selected. In the future the smoothing function will be included.
         - The average stellar mass also evolves due to sev. It accounts the proper solution, if a running ν is selected.
        3. **Tidal Effects**:
         - If tides are active, they contribute to mass loss and affect the half-mass radius evolution.
         - Finite escape time and rotation are also included in the tidal mass-loss calculation if selected. Rotation is implemented in a naive way and more study is needed.
         - In the future, the impact of tides on the average stellar mass will be included. Should affect low galactocentric distances and light star clusters mainly.
        4. **BH Mass Loss**:
         - BH mass loss includes mechanisms for ejection based on cluster parameters and equipartition states. Tidal effects are also available but not properly implemented, as the average BH mass remains unaffected. More study is needed.
        5. **Cluster Expansion**:
         - Expansion due to energy loss is modeled using Hénon's parameter (`zeta`). A constant value is assumed. In the future, a smoother connection will be made between the unbalanced and the balanced phase.
         - The contribution of evaporating stars to half-mass radius changes is also included, if the option is activated.
        6. **Combined Mass Loss**:
         - Total mass loss and its impact on the cluster size are calculated for both stars and BHs.
      
        - The returned derivatives (`Mst_dot`, `Mbh_dot`, `rh_dot`, `Mval_dot`, `r_dot`, `mst_dot`) can be used in numerical integrators to evolve the cluster state over time.
        """
        
        Mst = y[0] # [Msun] Mass of stars.
        Mbh = y[1] # [Mun] Mass of BHs.
        rh = y[2] # [pc] Half-mass radius.
        Mval = y[3] # Parameter for mass segregation.
        r = y[4] # Ratio rh / rv.
        mst = y[5] # [Msun] Average stellar mass.
        
        # Time instances are repeated multiple times. It is faster to assign them in local variables.        
        tcc = self.tcc  # [Myrs] Core collapse.
        tsev = self.tsev # [Myrs] Stellar evolution.
        tbh = self.t_bhcreation # [Myrs] Time instance when BHs have been created.

        mbh = self._mbh(Mbh)  # [Msun] Extract the average BH mass.
        if Mbh < mbh: Mbh, mbh = 0, 1e-99  # Useful statements for functions to be called later.
        if Mst < self.mst_inf: Mst, mst = 0, 1e-99 # Suggests that all stars have left the cluster
        if rh < 0: rh = 0 # Half-mass is strictly non-negative.
      
        # F = self.balance_dict[self.balance_model](t) # Future extension. Balancing function used to smoothly connect the two periods, (prior to post core collapse). With this, the if statements before and after tcc should be dropped.
        
        M = Mst + Mbh # [Msun] Total mass of the cluster. It is overestimated a bit initially because we assume Mbh > 0 from the start.
        fbh = Mbh / M # Fraction of BHs. Before core collapse is it overestimated. It is corrected after the first few Myrs.
        
    # These comments are to be activated (for now) only when E / M φ0 receives dependence on q = mbh / m.
    #    Np = M * ( fbh / mbh + (1 - fbh) / mst)  # Total number of particles.
    #    m = M / Np # [Msun] Total average mass.
        
        rt = self._rt(M) # [pc] Tidal radius.
        trh = self._trh(M, rh, fbh, mbh, mst) # [Myrs] Relaxation.
        trhstar = self._trhstar(M, rh, fbh, mbh, mst) if self.two_relaxations else trh  # [Myrs] Evaporation time scale. Check whether it is different from relaxation within rh.
       
        Mst_dot = rh_dot = Mbh_dot = 0 # [Msun / Myrs], [pc / Myrs], [Msun / Myrs] At first the derivatives are set equal to zero.
        Mval_dot = r_dot = mst_dot = 0 # [1 / Myrs], [1 / Myrs], [Msun / Myrs] Derivatives for parameter of mass segregation, ratio rh / rv and average stellar mass.
       
        zeta = self.zeta # Rate of energy emission each relaxation.
        G, x = self.G, self.x # Gravitational constant and exponent for finite escape time. Appear multiple times.
        kin, index = 1, 0 # Parameters that reappear. Used for the energy budget.
        
        # Add tidal mass loss.
        if self.tidal: # Check if we have tides.  
           
           xi = numpy.max(self.xi_function(rh, rt), self.fmin) # Tides. With F it should be xi = (self.fmin + F * (1 - self.fmin)) * self._xi(rh, rt). Now, it is compared to the fmin.
        
           if self.finite_escape_time:
              tcr = self._tcr(M, rh, k=self.tcross_index) # [Myrs] Crossing time.
              P = self.p * (trhstar / tcr) ** (1 - x)  # Check if we have finite escape time from the Lagrange points.
              xi *= P
              
           if self.rotation: # If the cluster rotates, the evaporation rate changes. A similar change to relaxation is used.
               xi *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (G * M)) ** self.gamma1 # Paramater gamma1 should be positive, since rotation makes it more difficult to reach the tail of the Maxwellian distribution. 
           
           # Effect of evaporating stars on the half mass radius. Keep only evaporation.   
           if self.escapers: # It is connected to the model for the potential of the cluster. 
              index = numpy.abs(self.cluster_model_function(rt, M)) / (G * M) # [pc^2 / Myrs^2 ] Get the value from the dictionary. It changes with respect to the potential assumed for the cluster.
              kin = self.kin if not self.varying_kin else 2 * (self._tcr(M, rh) * 0.138 / trhstar) ** (1 - x)  # Kinetic term for evaporating stars. Can be either a constant value or effectively a function of the number of particles.
           
           if Mst > 0: # Tidal mass loss appears only when stars are present.                 
              mst_dot += self.chi * (1 - self.m_breaks[0] / mst) * (1 - mst / self.mst_inf) * xi * mst / trhstar # [Msun / Myrs] Rate of change of average stellar mass due to tides.  
              Mst_dot -= xi * Mst / trhstar # [Msun / Myrs] Mass loss rate of stars due to tides (and central ejections).
              rh_dot += 6 * xi / r / trhstar * (1 - kin) * rh ** 2 * index * Mst / M # [pc / Myrs] If escapers carry negative energy as they leave, the half mass radius is expected to increase since it is similar to emitting positive energy. The effect is proportional to tides.
           
           # If the tidal field was important, an additional mass-loss mechanism would be needed.
           if self.dark_clusters and Mbh > 0 and t >= tbh: # Appears when BHs have been created.
              gbh = (exp(self.c_bh * fbh) - 1) / (exp(self.c_bh) - 1) # A simple dependence on fbh is shown. Trial model, will be studied extensively in the future.
              xi_bh = xi * gbh # Modify the tidal field that BHs feel such that it is negligible when many stars are in the cluster, and it starts increasing when only a few are left. 
              Mbh_dot -= xi_bh * Mbh / trhstar # [Msun / Myrs] Same description as stellar mass loss. Use relaxation in the outskirts. When BH eveporation becomes important, psi=1 approximately so the expressions should match.
             # mbh should increase with the tidal field. Another prescription is needed for mbh, or neglect it approximately. 
             # Now if BH evaporation is important and escapers is activated, it should be included here.
              rh_dot += 6 * xi_bh / r / trhstar * (1 - kin) * rh ** 2 * index * Mbh / M # [pc / Myrs] The kinetic term is the same here, it does not distinguish particle types.
        
        # Unbalanced phase. Check for the impact of the Virial radius on the half-mass expansion.
        if t < tcc:
            
            if self.Virial_evolution:
               r_dot += r * (self.rf - r) / trh # * F # [1 / Myrs] Because the Virial radius evolves slower compared to the half-mass radius, it accounts for this difference. Does not evolve in the balanced phase.
               rh_dot += rh * r_dot / r # Correction term from energy variation. A better expression for r_dot is needed.
       
            # Check if the cluster segregates.
            if self.mass_segregation:
               Mval_dot += Mval * (self.Mvalf - Mval) / trh  # [1 / Myrs] Evolution of parameter describing mass segregation. 
        
        # Balanced Phase. Check for stellar ejections.
        else:
             
            rh_dot += zeta * rh / trh # [pc / Myrs] Instead of the if statement, simply multiplying with F should work.
           
            S = max(self.a11 * self.a12 ** (3 / 2) * (Mbh / Mst) ** self.b1 * (mbh / mst) ** (self.b2 * 3 / 2), self.Sf) # Spitzer's parameter for equipartition. Use parameters to relate properties with rh to global ones.
            # Sf serves as a minimum value indicative of equipartion, below which the BH ejection rate is kept constant.
            
            beta_f = self.beta_function(S) # Factor used to modulate BH and stellar ejections. Depends on Spitzer´s parameter, and becomes constant when equipartition is reached.
        
            if Mbh > 0: # Check if BHs are present. BHs evolve only in the balanced phase and lose mass through ejections.
                  
               beta = self.beta # Ejection rate of BHs. Initially it is set as a constant. It should be changed when fbh increases a lot and the light component does not dominate anymore.
            
               if fbh < self.fbh_crit and self.running_bh_ejection_rate_1: # Condition for decreasing the ejection rate of BHs due to E / M φ0. A condition for mbh_crit may be needed.
                   beta *=  (fbh / self.fbh_crit) ** self.b4 #* (mbh / m / self.qbh_crit) ** self.b5 # A dependence on the average mass ratio mbh / m (or metallicity) may be needed.
               
               if self.running_bh_ejection_rate_2: # Decrease the ejection rate for clusters that are close to reaching equipartition.
                  
                  beta *= beta_f # Change the ejection rate with respect to Spitzer's parameter.
               
               Mbh_dot -= beta * zeta * M / trh  # [Msun / Myrs] Ejection of BHs each relaxation. 
           
            if Mst > 0: # A proper description should start with zero alpha_c, increase slightly up until core collapse, be a function of fbh, mbh and when we have no BHs it maximizes.
           
               alpha_c = self.alpha_ci # Initial ejection rate of stars. With F, it should be alpha_c += self.alpha_ci * F 
           
               if self.running_bh_ejection_rate_2: # Check if the ejection rate varies with the BH population.
                   alpha_c += (self.alpha_cf - alpha_c) * (1 - beta_f) # * F # This expression states that with many BHs, the ejection rate of stars should be small, and when the BHs vanish it increases. A similar rate of change as for the BH ejection rate is used. 
           
               if self.running_stellar_ejection_rate: # Check if the ejection rate decreases for increasing tides.
                   alpha_c *= (1 - 5 * xi / (3 * zeta)) # Alter alpha_c based on tides as introduced in EMACSS.
               
               Mst_dot -= alpha_c * zeta * M / trh # [Msun / Myrs] Contribution of central stellar ejections.
            
        rh_dot += 2 * Mst_dot / M * rh # [pc / Myrs] Impact of tidal mass loss to the size of the cluster.
       
        Mst_dotsev = 0 # [Msun / Myrs] Mass loss rate from stellar evolution.
               
        # Stellar mass loss. Impact of stellar winds.
        if self.sev and t >= tsev and Mst > 0: # This contribution is present only when Mst is nonzero.
            nu = numpy.max(self.nu_function(self.Z, t), 0) # Rate of stellar mass loss due to stellar winds. Taken from a dictionary, ensures that it is non-negative.
            mst_dot -= nu * (mst - mst * self.M0 / Mst * self.nu_factor) / t # [Msun/Myrs] When we consider stellar evolution, the average stellar mass changes through this differential equation. It is selected so that the case of a varying nu is properly described.
            Mst_dotsev -= nu * (Mst - self.M0 * self.nu_factor) / t  # [Msun / Myrs] Stars lose mass due to stellar evolution. If it is the sole mechanism for mass loss, it implies that N is constant since mst decreases with the same rate.
            rh_dot -= (Mval - 2) *  Mst_dotsev / M * rh # [pc / Myrs] The cluster expands for this reason. It is because a uniform distribution is assumed initially. 
        
        Mst_dot += Mst_dotsev  # [Msun / Myrs] Correct the total stellar mass loss rate. This way, resummation in rhdot is avoided.
   
        rh_dot += 2 * Mbh_dot / M * rh # [pc / Myrs] Contraction since BHs are removed.
    
        derivs = numpy.array([Mst_dot, Mbh_dot, rh_dot, Mval_dot, r_dot, mst_dot], dtype=object) # Save all derivatives in a sequence.

        return derivs # Return the derivatives in an array.

    # Extract the solution using the above differential equations.
    def _evolve(self, N, rhoh):
        """
        Simulates the dynamical evolution of a star cluster over time using the specified initial conditions and physical models.

        Parameters:
        -----------
        N : int
          Total number of stars in the cluster initially.
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

        def event(t, y):
            # Check if either condition is still valid.
            condition_1 = y[0] - self.Mst_min  # Positive if stars are above the threshold.
            condition_2 = y[1] - self.Mbh_min  # Positive if BHs are above the threshold.
 
            # Event function returns the maximum of the two conditions. Stops only when both are negative.
            return numpy.max([condition_1, condition_2]) # Maximum is chosen because it suggests that this population dominates.

        # Configure the event properties.
        event.terminal = True # Stop integration when the event is triggered.
        event.direction = -1 # Trigger the event from above. Means that the solution y is large initially and it stops when the difference becomes negative.

        # Solution.
        sol = solve_ivp(self._odes, [0, self.tend], y, method=self.integration_method, t_eval=self.t_eval, events=event, rtol=self.rtol, atol=self.atol, dense_output=self.dense_output, vectorized=self.vectorize) 

        self.t = sol.t / 1e3 # [Gyrs] Time.
        self.Mst = sol.y[0] # [Msun] Stellar mass.
        self.Mbh = sol.y[1] # [Msun] BH mass.
        self.rh = sol.y[2] # [pc] Half-mass radius.
        self.Mval = sol.y[3] # Parameter for segregation.
        self.r = sol.y[4] # Ratio rh / rv.
        self.mst = sol.y[5] # [Msun] Average stellar mass.
        
        self.mbh = numpy.array([self._mbh(x) for x in self.Mbh]) # [Msun] Average BH mass.
        self.Mbh = numpy.where(self.Mbh >= self.mbh, self.Mbh, 0) # [Msun] BH mass corrected for mbh > Mbh.
        self.mbh = numpy.where(self.Mbh >= self.mbh, self.mbh, 1e-99) # [Msun] Correct the average BH mass.
        
        self.Mst = numpy.where(self.Mst >= self.mst_inf, self.Mst, 0) # [Msun] Stellar mass corrected for low stellar masses.
        self.mst = numpy.where(self.Mst >= self.mst_inf, self.mst, 1e-99) # [Msun] Correct the average stellar mass.
        
        # Quantities for the cluster.
        self.M = self.Mst + self.Mbh # [Msun] Total mass of the cluster. BHs are included already. A small error in the first few Myrs is present.
        self.rt = self._rt(self.M) # [pc] Tidal radius.
        self.fbh = self.Mbh / self.M # BH fraction.
        self.mbh_max = numpy.array([self._find_mmax(x) for x in self.Mbh]) # [Msun] Maximum BH mass in the BHMF as a function of time. 
        self.mbh_max = numpy.where(self.Mbh >= self.mbh, self.mbh_max, 0) # Correct the upper BH mass to account for the case of no BHs left in the cluster.
        
        # Further properties.
        self.rv = self.rh / self.r # [pc] Virial radius.
        self.psi = self._psi(self.fbh, self.M, self.mbh, self.mst) # Friction term ψ.
        self.Np = numpy.round(self.M * ( self.fbh / self.mbh + (1 - self.fbh) / self.mst)).astype(int) # Number of components. 
        self.mav = self.M / self.Np # [Msun] Average mass of cluster over time, includes BHs. No significant change is expected given that Nbh <= O(1e3), apart from the beginning where the difference is a few percent.
        self.Nbh = numpy.round(self.Mbh / self.mbh).astype(int) # Number of BHs.
        self.E = - self.r * self.G * self.M ** 2 / (4 * self.rh) # [pc^2 Msun / Myrs^2] External energy of the cluster
        if (self.tidal): self.xi = numpy.maximum(self.xi_function(self.rh, self.rt), self.fmin) # Evaporation rate.
        self.trh = self._trh(self.M, self.rh, self.fbh, self.mbh, self.mst) # [Myrs] Relaxation within rh. Describes ejections.
        self.trhstar = self._trhstar(self.M, self.rh, self.fbh, self.mbh, self.mst) # [Myrs] Relaxation within rh. Describes evaporation.
        self.tcr = self._tcr(self.M, self.rh) # [Myrs] Crossing time.
        self.vesc = self._vesc(self.M, self.rh) # [km/s] Escape velocity. 
        self.S = self.a11 * self.a12 ** (3 / 2) * (self.Mbh / self.Mst) ** self.b1 * (self.mbh / self.mst) ** (3 / 2 * self.b2) # Parameter indicative of equipartition.
        self.phi_c = self.cluster_model_function(self.rt, self.M) # [pc^2/Myr^2] Cluster potential at the tidal radius.
        self.phi_0 = - (1.023 * self.vesc) ** 2 / 2 # [pc^2/Myr^2] Central potential. Needs fc to be correct.
        self.fcrun = self._fc(self.M, self.rh) # Parameter to estimate how the escape velocity evolves.
        self.beta_run = self.beta * self.beta_function(self.S) # Running beta factor.
        self.nu_run = numpy.maximum(self.nu_function(self.Z, sol.t), 0) # Running stellar mass loss rate.
        self.sigma = sqrt(0.2 * self.G * self.M / self.rh) # Velocity dispersion in the cluster.
        
        # Check if the results need to be saved. The default option is to save the solutions of the differential equations as well as the tidal radius and average masses of the two components. Additional inclusions are possible.
        if self.output:

            # Define table header
            table_header = "# t[Gyrs] Mbh[msun] Mst[msun] rh[pc] rt[pc] mbh[msun] mst[msun] mbh_max[msun]"

            # Prepare data for writing
            data = numpy.column_stack((self.t, self.Mbh, self.Mst, self.rh, self.rt, self.mbh, self.mst_sev, self.mbh_max))

            # Write data
            numpy.savetxt(self.outfile, data, header=table_header, fmt="%12.5e", comments="")


#######################################################################################################################################################################################################################################################
"""
Notes:
- Another set of values that can be used is:
    zeta, beta, n, Rht, ntrh, b, Mval0, Mval_cc, S0 = 0.0977, 0.0566, 0.7113, 0.0633, 0.1665, 2.14, 3.1156, 3.1156, 1.96.

- Functions _mstdot_ev, _balance and _fc are works in progress, currently deactivated or unused.
- self.Virial_evolution, self.finite_escape_time, self.rotation, self.dark_clusters and self.sev_Z_run are future statemens that require refinement.
- Towards small values of Mbh, the expected values of mbh, mbh_max do not coincide, the difference is a few solar masses.

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
