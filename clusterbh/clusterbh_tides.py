from __future__ import division
import numpy
import re
from pylab import log, sqrt, pi, log10, exp
from scipy.integrate import solve_ivp
import ssptools


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
        self.G  = 0.004499 # [pc^3 /Msun /Myr^2] Gravitational constant.
        self.Zsolar = 0.02 # Solar metallicity.
        
        # Cluster ICs.
        self.N = N # Initial number of stars.
        self.m0 = 0.606 # Taken from CMC models.
        self.fc = 1 # Factor for escape velocity. Different for other King models.
        self.rg = 8 # [kpc] Galactocentric distance of the cluster. For eccentric orbits, think of it as Ra(1 - e) where Ra is the apocenter, e is eccentricity.
        self.Z = 0.0002 # Metalicity of the cluster. Default to poor metal cluster. 
        
        # Model parameters.
        self.gamma = 0.02 # Parameter for Coulomb logarithm.
        self.x = 3./4 # Exponent used for finite escape time from Lagrange points.
        self.r = 0.8 # Ratio of the half-mass radius over the Virial radius initially.
        self.f = 1 # Prefactor that relates relaxation of stars with the half-mass relaxation.
        self.tcross_index = 1 # Prefactor for half-mass crossing time defined as tcr = index/sqrt(G ρ).
        self.c = 10 # Exponent for exponential tides.
        self.c_bh = 10 # Exponent for exponential tides used in the BH evaporation rate.
        self.rp = 0.8 # [kpc] Constant parameter for galactic potentials.
        self.omega0 = 0 # [1 / Myr] Angular frequency of cluster due to rotation, not orbit.
        self.gamma1 = 3 / 2 # Exponent for evaporation rate in the rotational part.
        self.a = 0 # Exponent for metallicity dependent stellar winds.
        self.c1 = 0 # First prefactor of time dependent stellar evolution.
        self.c2 = 0 # Second prefactor of time dependent stellar evolution.
        
        # Parameters that were fit to N-body / Monte Carlo models. All of them are dimensionless.
        self.zeta = 0.0977 # Energy loss per half-mass relaxation.
        self.beta = 0.0566 # Ejection rate of BHs from the core per relaxation.
        self.nu = 0.072 # Mass loss rate of stars due to stellar evolution. Here it is independent of metallicity, but a factor of (self.Z / self.Zsolar) ** self.a may be more appropriate.
        self.tsev = 1 # [Myrs]. Time instance when stars start evolving.
        self.a0 = 1 # Fix zeroth order in ψ.
        self.a11 = 1.94 # First prefactor in ψ. Relates the BH mass fraction within the half-mass to the total fraction.
        self.a12 = 0.634 # Second prefactor in ψ. relates the mass ratio of BHs over the total average mass within the half-mass radius and the total.
        self.a3 = 0.984 # Prefactor of average mass within the half-mass radius compared to the total.
        self.n = 0.7113 # Exponent in the power-law for tides.
        self.Rht = 0.0633 # Ratio of rh/rt to give correct Mdot due to tides. It is not necessarily the final value of rh/rt.
        self.ntrh = 0.1665 # Number of initial relaxations in order to compute core collapse instance.
        self.alpha_c = 0.0 # Ejection of stars from the center after core collapse.
        self.kin = 1 # Kinetic term of evaporating stars. Used to compute the energy carried out by evaporated stars. For CMC, kin = 2 * (log(self.gamma * N) / N) ** (1 - self.x)
        self.b = 2.14 # Exponent for parameter ψ. The choices are between 2 and 2.5, the former indicates the same velocity dispersion between components while the latter complete equipartition.
        self.b_min = 2 # Minimum exponent for parameter ψ.
        self.b_max = 2.26 # Maximum exponent for parameter ψ.
        self.b0 = 2 # Exponent for parameter ψ. It relates the fraction mst / m within rh and the total fraction.
        self.b1 = 1 # Correction to exponent of fbh in parameter ψ. It appears because the properties within the half-mass radius differ.
        self.b2 = 1.04 # Exponent of fraction mbh / m in ψ. Connects properties within rh with the global. 
        self.b3 = 0.96 # Exponent of average mass within rh.
        self.b4 = 0.17 # Exponent for the BH ejection rate. Participates after a critical value.
        self.b5 = 0.4 # Second exponent for the BH ejection rate. Participates after a critical value for the BH fraction.
        self.Mval0 = 3.1156 # Initial value for mass segregation. For a homologous distribution of stars, set it equal to 3.
        self.Mval_cc = 3.1156 # Contribution of stellar evolution in half-mass radius after core collapse. It is considered along with Henon's constant. If set equal to 2, it does not contribute.
        self.Mvalf = 4 # Final parameter for mass segregation.
        self.p = 0.1 # Parameter for finite time stellar escape from the Lagrange points. Relates escape time with relaxation and crossing time.
        self.fbh_crit = 0.005 # Critical value of the BH fraction to use in the ejection rate of BHs. Decreases the fractions E / M φ0.
        self.qbh_crit = 25 # [Msun] Ratio of mbh / m when the BH ejection rate starts decreasing.
        self.S0 = 1.96 # Parameter used for describing BH ejections when we are close to equipartition.
        self.gamma_exp = 7.75 # Parameter used for obtaining the correct exponent for parameter psi as a function of the BH fraction.
        
        # Some integration parameters.
        self.tend = 13.8e3 # [Myrs] Final time instance where we integrate to.
        self.dtout = 2 # [Myrs] Time step for integration.
        self.Mst_min = 100 # [Msun] Stop criterion.
        self.integration_method = "RK45" # Integration method.
        
        # Output.
        self.output = False # A Boolean parameter in order to save the results.
        self.outfile = "cluster.txt" # File to save the results if needed.

        # Conditions.
        self.kick = True # Condition to include natal kicks.   
        self.tidal = True # Condition to activate tides.
        self.escapers = False # Condition for escapers to carry negative energy as they evaporate from the cluster due to the tidal field.
        self.two_relaxations = True # Condition in order to have two relaxation time scales for the two components. Differentiates between ejections and evaporation.
        self.mass_segregation = False # Condition for mass segregation. If activated, parameter Mval evolves from Mval0 up until Mvalf within one relaxation.
        self.psi_exponent_run = False # Condition to have a running exponent in parameter ψ based on the BH fraction.
        self.finite_escape_time = False # Condition in order to consider escape from Lagrange points for stars which evaporate. Introduces an additional dependence on the number of particles on the tidal field.
        self.running_bh_ejection_rate_1 = True # Condition for decreasing the ejection rate of BHs due to E / M φ0.
        self.running_bh_ejection_rate_2 = True # Condition for decreasing the ejection rate of BHs due to equipartition.
        self.running_stellar_ejection_rate = False # Condition for changing the stellar ejection rate compared to the tidal field.
        self.rotation = False # Condition to describe rotating clusters.
        self.dark_clusters = False # Condition for describing dark clusters. When used, the approximations are not used.
        self.sev_t_run = False # Condition to have a running parameter nu with respect to time.
        self.sev_Z_run = False # Condition to have a running parameter nu with respect to metallicity.
        
        # Motion of cluster.
        self.Vc = 220. # [km/s] Circular velocity of singular isothermal galaxy. 
        
        # Galactic model.
        self.galactic_model = 'SIS'  # Default to 'SIS'.
        
        # Tidal model.
        self.tidal_model = 'Power_Law'  # Default to 'Power_Law'.
        
        # IMF. For top heavy, fine-tune a3.
        self.a_slope1 = -1.3 # Slope of mass function for the first interval.
        self.a_slope2 = -2.3 # Slope of mass function for the second interval.
        self.a_slope3 = -2.3 # Slope of mass function for the third interval.
        self.m_break1 = 0.08 # [Msun] Lowest stellar mass of the cluster.
        self.m_break2 = 0.5 # [Msun] Highest stellar mass in the first interval.
        self.m_break3 = 1. # [Msun] Highest stellar mass in the second interval.
        self.m_break4 = 150. # [Msun] Highest mass in the cluster.
        self.nbin1 = 5
        self.nbin2 = 5
        self.nbin3 = 20
        
        self.BH_IFMR = 'banerjee' # Default option for the IFMR.
        
        # Check input parameters. Afterwards we start computations.
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        # Define the tidal models dictionary.
        self.tidal_models = {
            'Power_Law': lambda rh, rt: 3 * self.zeta / 5 * (rh / rt / self.Rht) ** self.n, 
            'Exponential': lambda rh, rt: 3 * self.zeta / 5 * exp(self.c * rh / rt)
        }
        
        # Galactic Model.
        self.galactic_model_dict = {  # We create a dictionary for spherically symmetric potentials. A model can then be selected from the dictionary.
            'SIS': 2, # Singular isothermal sphere. It is the default.
            'Point_mass': 3, # Point mass galaxy.
            'Hernquist': (3 * self.rg + self.rp) / (self.rg + self.rp) ,  # Hernquist model.
            'Plummer': 3 * self.rg ** 2 / (self.rg ** 2 + self.rp ** 2),  # Plummer model.
            'Jaffe': (3 * self.rg + 2 * self.rp) / (self.rg + self.rp),   # Jaffe model.
            'NFW' : 1 + (2 * log(1 + self.rg / self.rp) - 2 * self.rg / (self.rg + self.rp) - (self.rg / (self.rg + self.rp)) ** 2 ) / (log(1 + self.rg / self.rp) - self.rg / (self.rg + self.rp)) , # Navarro-Frank-White model.
        }
        
        # Dictionary for the BH IFMR.
        self.BH_IFMR_dict = {
            'banerjee' : 'banerjee20',
            'cosmic': 'cosmic-rapid'
        }
        
        # Check the available IMF. The default option is 3 slopes, bins and break masses.
        self.a_slopes, self.m_breaks, self.nbins = self._IMF_read() # The user is allowed to include other regions in the IMF, for instance m > 150 with a steeper slope, -2.7.
        
     #   self.m0 = self._initial_average_mass(self.a_slopes, self.m_breaks) # Average mass obtained for this particular IMF. It is 3% off compared to M / N at t=0 in the CMC models.
        
        self.FeH = log10(self.Z / self.Zsolar) # Metallicity in solar units.
        
        self.M0 = self.m0 * N # [Msun] Total mass of stars (cluster) initially.
        self.rh0 = (3 * self.M0 / (8 * pi * rhoh)) ** (1./3) # [pc] Half-mass radius initially.
        
        self.vesc0 = self._vesc(self.M0, self.rh0) # [km/s] Initial escape velocity.
        
        # Check the IFMR model used in the SSP tools.
        if hasattr(self, 'BH_IFMR') and self.BH_IFMR in self.BH_IFMR_dict:
            self.BH_IFMR_method = self.BH_IFMR_dict[self.BH_IFMR]
        
        # Implement kicks, if activated, for this IMF, number of stars, with such metallicity and central escape velocity.
        self.ibh = ssptools.InitialBHPopulation.from_powerlaw(self.m_breaks, self.a_slopes, self.nbins, self.FeH, N0=N, vesc=self.vesc0, natal_kicks=self.kick, BH_IFMR_method=self.BH_IFMR_method) # Version 2 of SSP tools.
        
        self.Mbh0 = self.ibh.Mtot  # [Msun] Expected initial mass of BHs due to kicks.
        self.f0 = self.Mbh0 / self.M0 # Initial fraction of BHs. It should be close to 0.05 for poor-metal clusters. 
        self.Nbh0 = numpy.round(self.ibh.Ntot).astype(int) # Initial number of BHs. We round the number.
        self.mbh0 = self.Mbh0 / self.Nbh0 # [Msun] Initial average BH mass.
        self.Mst_lost = self.ibh.Ms_lost # [Msun] Mass of stars lost in order to form BHs.
        self.t_bhcreation = self.ibh.age # [Myrs] Time needed to form these BHs.
        
        # Initial relaxation is extracted irrespective of the BH population. It is an approximation.
        self.psi0 = self._psi(self.Mbh0 / (self.M0 + self.Mbh0 - self.Mst_lost), self.M0 + self.Mbh0 - self.Mst_lost, self.mbh0, self.t_bhcreation) # Use as time instance the moment when all BHs are formed. In this regime, tides are not so important so all the mass-loss comes from stellar evolution. Applies if the cluster has not collapsed.
        self.trh0 = 0.138 / (self.m0 * self.psi0 * log(self.gamma * self.N)) * sqrt(self.M0 * self.rh0 ** 3 / self.G ) # [Myrs] We use m0 since at t=0 the mass distribution is the same everywhere.
        self.tcc = self.ntrh * self.trh0 # [Myrs] Core collapse. Depends on metallicity, size and Mass of the cluster.
        
        # Check if the galactic model has changed. Based on the available choices in the dictionary, change the value accordingly.
        if hasattr(self, 'galactic_model') and self.galactic_model in self.galactic_model_dict:
            self.rt_index = self.galactic_model_dict[self.galactic_model]

        self.evolve(N, rhoh)
    
    def _IMF_read(self):
        """
        Processes and validates class attributes related to the initial mass function (IMF).

        - Extracts `a_slopeX`, `m_breakX`, and `nbinX` attributes dynamically. The default number are 3, 4, 3 but the user is allowed to insert more. They should be in a sequence.
        - Sorts these attributes by their numeric suffix.
        - Validates `m_breakX` to ensure values are strictly increasing, stopping at the first invalid entry.
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
        self.max_m_break = valid_m_breaks[-1]
        num_valid_m_breaks = len(valid_m_breaks)
        num_slopes_bins = num_valid_m_breaks - 1

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
          Breakpoints of the mass ranges in solar masses. The length of this list should be one more than `a_slopes`.

       Returns:
       --------
       float
        The average mass of stars calculated using the IMF and its normalization.
    
       Notes:
       ------
       - This function integrates the IMF.
       - The function assumes the IMF is continuous across the mass breakpoints.
       - Special handling is included for slopes to avoid division by zero.
       """
       def integrate_m_p(m_low, m_high, p):
        
           if p == -1:  # Special case where p = -1 to avoid division by zero
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
         
       normalization_constants.insert(0, c_values[0])
       
       stellar_mass = 0
       stellar_number = 0
       
       for i, a in enumerate(a_slopes):
          m_low = m_breaks[i]
          m_high = m_breaks[i + 1]
          c_i = normalization_constants[i]
          
          # Compute numerator 
          stellar_mass += c_i * integrate_m_p(m_low, m_high, a + 1)
         
          # Compute denominator
          stellar_number += c_i * integrate_m_p(m_low, m_high, a)
    
       # Calculate the average mass
       return stellar_mass / stellar_number
    
    
    # Function to determine by how much the BHMF changes, given a particular mass loss.
    def _deplete_BHMF(self, M_eject, M_BH, N_BH):
        """
        Deplete the BHMF by ejecting mass starting from the heaviest bins.
    
        Parameters:
         -----------
        M_eject : float
           Total mass to eject from the BHMF, in solar masses (Msun).
        M_BH : numpy.ndarray
           Array representing the total mass in each mass bin, in solar masses (Msun).
        N_BH : numpy.ndarray
           Array representing the number of BHs in each mass bin.

        Returns:
        --------
        tuple of numpy.ndarray
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
    
    # Average BH mass.
    def _mbh(self, Mbh):
        """
        Calculate the updated average BH mass after ejecting a specified amount of BH mass.
    
        Parameters:
        -----------
        Mbh : float
          Total BH mass after ejection, in solar masses (Msun).

        Returns:
        --------
        float
          Average BH mass after ejection, in solar masses (Msun).
    
        Notes:
        ------
        - The function first calculates the total mass to be ejected (`M_eject`) in [Msun].
        - It then updates the BHMF by removing mass using `_deplete_BHMF`.
        - Finally, the average mass of the remaining BHs is computed by dividing the total mass by the total number. Units in [Msun].
        """
        # Determine amount of BH mass (total) to eject.
        M_eject = self.Mbh0 - Mbh # [Msun]

        # Deplete the initial BH MF based on this M_eject.
        M_BH, N_BH = self._deplete_BHMF(M_eject, self.ibh.M, self.ibh.N) # [Msun, dimensionless]
        return M_BH.sum() / N_BH.sum() # [Msun]
       
    # Tidal radius. This is the expression for SIS. For other gravitational potentials, other expressions may be needed.
    def _rt(self, M): 
        """
        Calculate the tidal radius of a star cluster based on its mass and orbital properties.

        Parameters:
        -----------
        M : float
           Total mass of the cluster, in solar masses (Msun).

        Returns:
        --------
        float
           Tidal radius of the cluster, in [pc].

        Notes:
        ------
        - The tidal radius depends on the cluster's mass (`M`) [Msun], the angular velocity squared (`O2`) [1/Myrs^2], and a scaling parameter (`rt_index`) differing between galactic potentials.
        - The formula assumes a spherically symmetric potential, applicable to both circular and eccentric orbits (using an effective distance for the latter).
        """
        
        # Angular velocity squared.
        O2 = (self.Vc * 1.023 / (self.rg * 1e3)) ** 2 # [ 1 / Myrs^2]
        
        # Parameter for circular orbits in the tidal radius. The prescription is the same regardless of the orbit. Eccentric orbits can be described as circular using an effective distance.
        rt_index = self.rt_index # The value is extracted from the dictionary.
      
        # Tidal radius.
        rt = (self.G * M / (rt_index * O2)) ** (1./3) # [pc] The expression is valid as long as the potential is spherically symmetric, regardless of the orbit.
        return rt

    # Average mass of stars, due to stellar evolution.    
    def _mst(self, t):
        """
        Calculate the average stellar mass of the cluster over time.

        Parameters:
        -----------
        t : float
           Time elapsed since the beginning of the cluster's evolution, in [Myrs].

        Returns:
        --------
        float
           Average stellar mass of the cluster, in solar masses [Msun].

        Notes:
        ------
        - For times greater than the stellar evolution timescale (`tsev`) [Myrs], the stellar mass evolves following a power-law decay.
        - If the time is less than or equal to the stellar evolution timescale, the average stellar mass remains constant (`m0`) [Msun].
        - This calculation does not account for tidal effects; to include them, `mst` would need to be derived from differential equations.
        """
    
        mst = self.m0 * (t / self.tsev) ** (- self.nu) if t > self.tsev else self.m0 # [Msun]
        return mst
        
    # Friction term ψ in the relaxation. It is characteristic of mass spectrum, here due to BHs. We neglect O(2) corrections.
    def _psi(self, fbh, M, mbh, t): 
        """
        Calculate psi for the cluster based on various parameters.

        Parameters:
        -----------
        fbh : float
           A factor representing the BH fraction in the range [0, 1].
    
        M : float
           Total mass of the cluster, in solar masses [Msun].
    
        mbh : float
           Average BH mass in solar masses (Msun).
    
        t : float
           Time elapsed since the beginning of the cluster's evolution, in [Myrs].

        Returns:
        --------
        float
           The calculated psi value, which depends on the BH fraction, cluster mass,
           average masses and time.

        Notes:
        ------
        - If the BH mass is less than the average BH mass (mbh), 
          it indicates that BHs have been ejected, and the function returns a constant value `a0` for psi.
        - The number of particles `Np` is calculated using the `_N` function, which depends on the cluster's mass, 
          the BH fraction, and the elapsed time.
        - The average stellar mass `mav` is derived by dividing the total cluster mass by the number of particles.
        - The stellar mass `mst` is obtained from the `_mst` function, which gives the average stellar mass at time `t`.
        - The exponent `gamma` is either a constant value `b` or it evolves based on the BH fraction (if `psi_exponent_run` is true),
          starting from an equal velocity dispersion and evolving towards equipartition as time passes.
        - The expression for `psi` includes a combination of parameters like `a0`, `a11`, `a12`, `b1`, `b2`, and others, 
          which relate the properties within the cluster's half-mass radius (`rh`) to the global properties of the cluster.
        - For dark clusters (`dark_clusters` is true), a more complex expression for `psi` is used, 
          incorporating contributions from both the stellar and BH populations.
        """
        if M * fbh < mbh : return self.a0 # This means that BHs have been ejected so we have no psi.
        
        # Number of particles.
        Np = self._N(M, fbh, mbh, t)
       
        # Average mass and stellar mass.
        mav = M / Np # [Msun]
        mst = self._mst(t)  # [Msun] Neglect this if mst is obtained from differential equation like EMACSS.
        
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
    
    # Number of particles, approximately stars. We write it as a function of the total mass and the BH fraction. The number of BHs is a minor correction since typically 1 star in 1e3 becomes a BH, so clusters with 1e6 stars have 1e3 BHs roughly speaking, a small correction.    
    def _N(self, M, fbh, mbh, t):
        """
        Calculate the number of particles in the cluster, including both black holes (BHs) and stars.

        Parameters:
        -----------
        M : float
           Total mass of the cluster, in solar masses [Msun].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass in solar masses [Msun].
    
        t : float
           Time elapsed since the beginning of the cluster's evolution, in [Myrs].

        Returns:
        --------
        float
          The total number of particles in the cluster, considering both BHs and stars.

        Notes:
        ------
        - The number of BHs is computed by dividing the total BH mass (`M * fbh`)
          by the average BH mass (`mbh`). Both are in [Msun].
        - If the BH mass (`M * fbh`) is less than the average BH mass (`mbh`), the black hole fraction is set to zero.
        - The number of stars is calculated by dividing the remaining mass (`M * (1 - fbh)`) by the average stellar mass (`mst`), which is obtained from the `_mst` function.
        """
        # Number of particles.
        Np = 0
        
        # Include BHs if we have them.
        if M * fbh < mbh : fbh = 0
        
        Np += M * fbh / mbh 

        # Now we consider the number of stars.
        Np += M *  (1 - fbh) / self._mst(t)  # Here it should be mst, that takes into account changes from tides. Should be important at late times for small galactocentric distances only.
        return Np

    # Relaxation as defined by Spitzer. Here we consider the effect of mass spectrum due to BHs. 
    def _trh(self, M, rh, fbh, mbh, t):
        """
        Calculate the relaxation timescale (`trh`) for a cluster, taking into account mass, radius, 
        BH fraction and time evolution.

        Parameters:
        -----------
        M : float
          Total mass of the cluster, in solar masses [Msun].
    
        rh : float
           Half-mass radius of the cluster, in parsecs [pc].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass in solar masses [Msun].
    
        t : float
          Time elapsed since the beginning of the cluster's evolution, in [Myrs].

        Returns:
        --------
        float
           The relaxation timescale (`trh`) of the cluster, in [Myrs].

        Notes:
        ------
        - If the total mass `M` or the half-mass radius `rh` is less than or equal to zero, the function returns a very small value (`1e-99`).
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass within `rh` is computed using a power-law fit based on the total mass and the number of particles.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, 
          and the gravitational constant `G`. The timescale is further modified by the cluster's `psi` value, which depends on the BH fraction.
        - If the cluster undergoes rigid rotation (`self.rotation` is `True`), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model assuming constant rotation. The expression by King is used.
        """
        
        if M <= 0 or rh <= 0: return 1e-99
        
        # Number of particles.
        Np = self._N(M, fbh, mbh, t) 
       
        # Average mass within rh. We use a power-law fit.
        mav =  self.a3 * (M / Np) ** self.b3  # [Msun]
        
        # Relaxation.
        trh = 0.138 * sqrt(M * rh ** 3 / self.G) / (mav * self._psi(fbh, M, mbh, t) * log(self.gamma * Np)) # [Myrs]
        
        if (self.rotation): # Effect of rigid rotation.
            trh *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** (3 / 2) # Assume constant rotation for now.
        
        return trh
        
    # Relaxation for stars depending on the assumptions. 
    def _trhstar(self, M, rh, fbh, mbh, t): 
        """
        Calculate the evaporation timescale (`trh`) for a cluster, taking into account mass and radius.

        Parameters:
        -----------
        M : float
          Total mass of the cluster, in solar masses [Msun].
    
        rh : float
           Half-mass radius of the cluster, in parsecs [pc].
    
        fbh : float
           The BH fraction in the range [0, 1].
    
        mbh : float
           Average BH mass in solar masses [Msun].
    
        t : float
          Time elapsed since the beginning of the cluster's evolution, in [Myrs].

        Returns:
        --------
        float
           The evaporation timescale (`trhstar`) of the cluster, in [Myrs].

        Notes:
        ------
        - If the total mass `M` or the half-mass radius `rh` is less than or equal to zero, the function returns a very small value (`1e-99`).
        - The number of particles `Np` is computed using the `_N` function, which accounts for both stars and BHs in the cluster.
        - The average mass within `rh` is computed using a power-law fit based on the total mass and the number of particles.
        - The relaxation timescale is calculated using a formula that depends on the total mass `M`, the half-mass radius `rh`, the average mass `mav`, 
          and the gravitational constant `G`. 
        - If one relaxation time scale is used (`self.two_relaxations` is `False`), the function is set equal to `trh`.
        - If the cluster undergoes rigid rotation (`self.rotation` is `True`), the relaxation timescale is adjusted to account for the effect of rotation, 
          using a simplified model assuming constant rotation.
        """
        
        if M <= 0 or rh <= 0: return 1e-99
        
        # Half mass relaxation.
        trh = self._trh(M, rh, fbh, mbh, t) # [Myrs]
         
        if not (self.two_relaxations): return trh # In case a model uses only one relaxation time scale.
        
        # Number of particles.
        Np = self._N(M, fbh, mbh, t)
        
        # Use the average mass of the whole cluster to compute this relaxation timescale.
        mav = M / Np #[ Msun]
        
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
           Total mass of the cluster, in solar masses [Msun].
    
        rh : float
           Half-mass radius of the cluster, in parsecs [pc].

        Returns:
        --------
        float
           The crossing timescale (`tcr`) of the cluster, in [Myrs].

        Notes:
        ------
        - If the total mass `M` or the half-mass radius `rh` is less than or equal to zero, the function returns a very small value (`1e-99`).
        - The prefactor `k` is derived from the expression `tcr = 1 / sqrt(G * rhoh)`, where `rhoh` is the mass density within the half-mass radius.
          The value of `k` is adjusted by the `tcross_index` parameter, which can modify the crossing time based on specific cluster properties.
        - The crossing timescale is calculated using the formula: `tcr = k * sqrt(rh^3 / (G * M))`, where `rh` is the half-mass radius and `M` is the total mass of the cluster.
        """
        
        if M <= 0 or rh <= 0: return 1e-99
        # Prefactor derived from the expression tcr = 1 / sqrt(G rhoh). If the numerator is not 1, it needs to be changed trhough tcross_index.
        if tcross_index is None:
           tcross_index = self.tcross_index
        k = 2 * sqrt(2 * pi / 3) * tcross_index
         
        tcr = k * sqrt(rh ** 3 / (self.G * M)) # [Myrs]
        return tcr
    
    # Escape velocity.
    def _vesc(self, M, rh):
        """
        Calculate the escape velocity (`vesc`) for a cluster, based on its total mass and half-mass radius.

        Parameters:
        -----------
        M : float
           Total mass of the cluster, in solar masses [Msun].
    
        rh : float
           Half-mass radius of the cluster, in parsecs [pc].

        Returns:
        --------
        float
           The escape velocity (`vesc`) at the center of the cluster, in [km/s].

        Notes:
        ------
        - The density `rhoh` is computed as the mass enclosed within the half-mass radius (`rh`), using the formula: 
          `rhoh = 3 * M / (8 * pi * rh^3)`. The density is in units of [Msun/pc^3].
        - The escape velocity `vesc` is then calculated using the relation: 
          `vesc = 50 * (M / 1e5)^(1/3) * (rhoh / 1e5)^(1/6)`, where the units for `vesc` are in [km/s]. 
          This formula expresses the central escape velocity based on the mass and the density of the cluster.
        - The escape velocity is further augmented by multiplying it by the factor `fc`, which adjusts the value according to the specific King model used for the cluster.
        """
        
        # Density .
        rhoh = 3 * M / (8 * pi * rh ** 3) # [Msun / pc^3]
        
        # Escape velocity.
        vesc = 50 * (M / 1e5) ** (1./3) * (rhoh / 1e5) ** (1./6) # [km/s] Central escape velocity as a function of mass and density.
        vesc *= self.fc # Augment the value for different King models.
        return vesc
        
    # Tides.
    def _xi(self, rh, rt):
        """
        Calculate the tidal parameter (`xi`) for a cluster, which describes the influence of tides
        based on the ratio of the half-mass radius to the tidal radius.

        Parameters:
        -----------
        rh : float
           Half-mass radius of the cluster, in parsecs [pc].
    
        rt : float
          Tidal radius of the cluster, in parsecs [pc].

        Returns:
        --------
        float
          The tidal parameter (`xi`), dimensionless.

        Notes:
        ------
        - The function calculates the tidal parameter using the respective expression from the dictionary. The default option is a power-law.
        """
        
        xi = 0
        if self.tidal_model in self.tidal_models:
            xi = self.tidal_models[self.tidal_model](rh, rt)
        else:
            raise ValueError(f"Invalid tidal model: {self.tidal_model}.") 
        return xi 
    
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
           Time elapsed since the beginning of the cluster's evolution, in [Myrs].

        Returns:
        --------
        float
           The mass-loss rate parameter (`nu`), which may vary with metallicity and time.

        Notes:
        ------
        - The function computes the mass-loss rate parameter `nu` using a default value (`self.nu`), and it introduces a time-dependent factor (`f`) 
          based on a quadratic logarithmic dependence on time. A future extension will introduce a dictionary so the user can select from the available dependences.
        - If the `sev_Z_run` flag is set to `True`, the mass-loss rate is further adjusted for metallicity (`Z`). This dependence allows for the effect of metallicity on mass loss, with `self.Zsolar` being the solar metallicity.
        - If the `sev_t_run` flag is set to `True` and the time-dependent factor `f` is positive, `nu` is multiplied by `f`, further modifying the mass-loss rate over time.
        """
        
        # Introduce a dependence of metallicity and time for stellar mass-loss rate. If activated.
        nu = self.nu
        
        if (self.sev_Z_run):
            nu *= (Z / self.Zsolar) ** self.a # Another dependence on metallicity may be better.
        if (self.sev_t_run):
            f = (1 + self.c1 * log(t) - self.c2 * log(t) ** 2) # Time dependence for parameter nu.
            if f > 0:
               nu *= f # Check if the expression is positive. In any other case, set it equal to 0.
            else: return 0
        return nu
    
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

        Returns:
        --------
        numpy.array
           An array containing the time derivatives of the state variables:
           - `Mst_dot`: Stellar mass loss rate [Msun/Myr].
           - `Mbh_dot`: Black hole mass loss rate [Msun/Myr].
           - `rh_dot` : Rate of change of the half-mass radius [pc/Myr].
           - `Mval_dot` : Evolution of the mass segregation parameter [1/Myr].

        Notes:
        ------
        - This function models the dynamical evolution of the star cluster using multiple physical processes.
        - The calculations include:
        1. **Core Collapse**:
         - After core collapse time (`tcc`), the parameter for mass segregation, `Mval`, transitions to a constant value.
        2. **Stellar Evolution**:
         - Stellar mass loss due to stellar evolution is modeled using mass-loss rate (`nu`). Function `_nu` is used.
         - Mass segregation is evolved if the option is selected.
        3. **Tidal Effects**:
         - If tides are active, they contribute to mass loss and affect the half-mass radius evolution.
         - Finite escape time and rotation are also included in the tidal mass-loss calculation if selected.
        4. **BH Mass Loss**:
         - BH mass loss includes mechanisms for ejection based on cluster parameters and equipartition states. Tidal effects are also available.
        5. **Cluster Expansion**:
         - Expansion due to energy loss is modeled using Henon's parameter (`zeta`). A constant value is assumed.
         - The contribution of evaporating stars to half-mass radius changes is also included, if the option is activated.
        6. **Combined Mass Loss**:
         - Total mass loss and its impact on the cluster size are calculated for both stars and BHs.
      
        - The returned derivatives (`Mst_dot`, `Mbh_dot`, `rh_dot`, `Mval_dot`) can be used in numerical integrators to evolve the cluster state over time.
        """
        
        Mst = y[0] # [Msun] Mass of stars.
        Mbh = y[1] # [Mun] Mass of BHs.
        rh = y[2] # [pc] Half-mass radius.
        Mval = y[3] # Parameter for mass segregation.
         
        tcc = self.tcc  # [Myrs] Core collapse.
        tsev = self.tsev # [Myrs] Stellar evolution.

        mbh = self._mbh(Mbh)  # [Msun] Extract the average BH mass.
        if Mbh < mbh : Mbh = 0 # It would be unphysical to have a BH mass that is lesser than the average BH mass. This happens only for the final BH of course.
        mst = self._mst(t) # [Msun] Average stellar mass.
        
        M = Mst + Mbh # [Msun] Total mass of the cluster. It overestimates a bit initially because we assume Mbh > 0 from the start.
        fbh = Mbh / M # Fraction of BHs. Before core collapse is it overestimated
        S = self.a11 * self.a12 ** (3/2) * (Mbh / Mst) ** self.b1 * (mbh / mst) ** (3 / 2 * self.b2) # Spitzer's parameter for equipartition.
        
    #    Np = self._N(M, fbh, mbh, t) # Total number of particles.
    #    m = M / Np # [Msun] Total average mass.
        
        rt = self._rt(M) # [pc] Tidal radius.
        trh = self._trh(M, rh, fbh, mbh, t) # [Myrs] Relaxation.
        trhstar = self._trhstar(M, rh, fbh, mbh, t) # [Myrs] Evaporation time scale.
        tcr = self._tcr(M, rh) # [Myrs] Crossing time.
      
        xi = 0 # Tides are turned off initially. We build them up if needed.
        alpha_c = 0 # Ejection rate of stars from the center are set equal to zero.
        
        Mst_dot, rh_dot, Mbh_dot = 0, 0, 0 # [Msun / Myrs], [pc / Myrs], [Msun / Myrs] At first the derivatives are set equal to zero, then we build them up.
        Mval_dot = 0 # [1 / Myrs] Derivative for parameter of mass segregation.
       
        M_val = Mval # This parameter describes mass segregation, used in the equation for rh. The reason why it changes to a constant value after core collapse is because of Henon's statement, here described by parameter zeta.
        if t >= tcc:
            M_val = self.Mval_cc # After core collapse, a constant value is assumed, different. In the isolated version of clusterBH the value is 1. For Mval_cc = 2, no contribution is assumed.
                  
        Mst_dotsev = 0 # [Msun / Myrs] Mass loss rate from stellar evolution.
               
       # Stellar mass loss.
        
        if t >= tsev and Mst > 0: # This contribution is present only when Mst is nonzero.
            nu = self._nu(self.Z, t)
            Mst_dotsev -= nu * Mst / t # [Msun / Myrs] Stars lose mass due to stellar evolution.
            rh_dot -= (M_val - 2) *  Mst_dotsev / M * rh # [pc / Myrs] The cluster expands for this reason. It is because we assume a uniform distribution initially. 
           
            if t < tcc and (self.mass_segregation):
               Mval_dot += Mval * (self.Mvalf - Mval) / trh  # [1 / Myrs] Evolution of parameter describing mass segregation. 
            
        # Add tidal mass loss.
        
        if (self.tidal): # Check if we have tides.  
            
           xi += self._xi(rh, rt) # Tides.
           if (self.finite_escape_time): 
              P = self.p * (trhstar / tcr) ** (1 - self.x)  # Check if we have finite escape time from the Lagrange points.
              xi *= P
              
           if (self.rotation): # If the cluster rotates, the evaporation rate changes. A similar change to relaxation is used.
               xi *= (1 - 2 * self.omega0 ** 2 * rh ** 3 / (self.G * M)) ** self.gamma1
             
        if t >= tcc and Mst > 0: 
           alpha_c += self.alpha_c # Central stellar ejection rate. If we were to include ejections before tcc, this expression needs to be modified. 
           if (self.running_stellar_ejection_rate):
               alpha_c *= (1 - 5 * xi / 3) # We may alter alpha_c based on tides as introduced in EMACSS.
               
      # Expansion due to energy loss. It is due to the fact that the cluster has a negative heat capacity due to its gravitational nature.
        if t >= tcc:
           rh_dot += self.zeta * rh / trh # [pc / Myrs] 
        
        if Mst > 0:  # Tidal mass loss and stellar ejections appears only when we have stars.                 
           Mst_dot -= xi * Mst / trhstar + alpha_c * self.zeta * M / trh # [Msun / Myrs] Mass loss rate of stars due to tides (and central ejections).
      
        rh_dot += 2 * Mst_dot / M * rh # [pc / Myrs] Impact of tidal mass loss to the size of the cluster.
        
        Mst_dot += Mst_dotsev  # [Msun / Myrs] Correct the total stellar mass loss rate.
   
        # Effect of evaporating stars on the half mass radius. Keep only evaporation, not ejections here so use xi and not xi_total.   
        if (self.escapers) and (self.tidal): # It is connected to the model for the potential of the cluster. Here, point-mass is used but the result differs if for instance a Plummer model is used. 
           rh_dot += 6 * xi / self.r / trhstar * (1 - self.kin) * rh ** 2 / rt * Mst / M # [pc / Myrs] If escapers carry negative energy as they leave, the half mass radius is expected to increase since it is similar to emitting positive energy. The effect is proportional to tides.
           # If φ(rt)=-3/2 GM/sqrt(rt**2 + b**2) for Plummer, the above expression needs sqrt(rt**2 + b**2) in the denominator, instead of rt.
           
        if Mbh > 0 and t >= tcc: # Check if we have BHs so that we can evolve them as well.
              
           beta = self.beta # Ejection rate of BHs. Initially it is set as a constant.
           if fbh < self.fbh_crit and (self.running_bh_ejection_rate_1): # Condition for decreasing the ejection rate of BHs due to E / M φ0. # A condition for mbh_crit may be needed.
               beta *=  (fbh / self.fbh_crit) ** self.b4 #* (mbh / m / self.qbh_crit) ** self.b5 # A dependence on average mass (or metallicity) may be needed.
           
           if (self.running_bh_ejection_rate_2): # Decrease the ejection rate for clusters that are close to reaching equipartition.
              beta *= 1 - exp( - S / self.S0) # Another ansatz with S-dependence may be chosen here. However, for large S we should have a constant beta and for vanishing S, beta vanishes as well.
           
           Mbh_dot -= beta * self.zeta * M / trh  # [Msun / Myrs] Ejection of BHs each relaxation. 
       
           # If the tidal field was important, an additional mass-loss mechanism would be needed.
           if (self.dark_clusters):
              gbh = (exp(self.c_bh * fbh) - 1) / (exp(self.c_bh) - 1) # A simple dependence on fbh is shown.
              xi_bh = xi * gbh # Modify the tidal field that BHs feel such that it is negligible when we have many stars, and it starts increasing when we have a few only. 
              Mbh_dot -= xi_bh * Mbh / trhstar # [Msun / Myrs] Same description as stellar mass loss. Use relaxation in the outskirts.
             
              # Now if this is important and escapers is activated, it should be included here
              if (self.escapers) : # Currently it is not applied to BHs if the tidal field affects them.
                 rh_dot += 6 * xi_bh / self.r / trhstar * (1 - self.kin) * rh ** 2 / rt * Mbh / M # [pc / Myrs] 
                  
           rh_dot += 2 * Mbh_dot / M * rh # [pc / Myrs] Contraction since BHs are removed.
    
        derivs = numpy.array([Mst_dot, Mbh_dot, rh_dot, Mval_dot], dtype=object) # Save all derivatives in a sequence.

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
        - Implements an event to terminate the simulation when the stellar mass drops below a minimum threshold (`Mst_min`).

        Output:
        ------
        - Cluster parameters are computed and can be saved in a text file if selected.
        
        Notes:
        ------
        - Initial conditions for stellar mass, BH mass, half-mass radius, and segregation parameter are set to `self.M0`, `self.Mbh0`, `self.rh0`, and `self.Mval0`, respectively.
        - Results are stored as attributes of the object for further analysis.
        """
        
        Mst = [self.M0] # [Msun] Initial stellar mass.
        Mbh = [self.Mbh0] # [Msun] Initial BH mass.
        rh = [self.rh0] # [pc] Initial half-mass radius.
        Mval = [self.Mval0] # Initial parameter for mass segregation.

        y = [Mst[0], Mbh[0], rh[0], Mval[0]] # Combine them in a multivariable.

        def Mst_min_event(t, y):  # Event in order to stop when stars are lost.
            return y[0] + y[1] - self.Mst_min

        Mst_min_event.terminal = True # Find solutions as long as the event holds.
        Mst_min_event.direction = -1 # Stop when we find a time instance such that the event is no longer True.

        t_eval = numpy.arange(0, self.tend, self.dtout) if self.dtout is not None else None # [Myrs] Time steps to which the solution will be computed.
        
        # Solution.
        sol = solve_ivp(self.odes, [0, self.tend], y, method=self.integration_method, t_eval=t_eval) # rtol=1e-8, atol=1e-10 # Options for increased accuracy.

        self.t = numpy.array([x / 1e3 for x in sol.t]) # [Gyrs] Time.
        self.Mst = sol.y[0] # [Msun] Stellar mass.
        self.Mbh = sol.y[1] # [Msun] BH mass.
        self.rh = sol.y[2] # [pc] Half-mass radius
        self.Mval = sol.y[3] # Parameter for segregation
       
        self.mbh = numpy.array([self._mbh(x) if y > self.tcc else self.mbh0 for (x, y) in zip(self.Mbh, sol.t)]) # [Msun] Average BH mass.
        self.Mbh = numpy.array([x if x >= y else 0 for x, y in zip(self.Mbh, self.mbh)]) # [Msun] BH mass corrected for mbh > Mbh.
        self.mbh = numpy.array([y if x >= y else 0 for x, y in zip(self.Mbh, self.mbh)]) # [Msun] Correct the average BH mass.
        
        # Quantities for the cluster.
        self.M = self.Mst + self.Mbh # [Msun] Total mass of the cluster. We include BHs already.
        self.rt = self._rt(self.M) # [pc] Tidal radius.
        self.fbh = self.Mbh / self.M # BH fraction.
        
        self.psi = numpy.array([self._psi(x, y, z, u) for (x, y, z, u) in zip(self.fbh, self.M, self.mbh, sol.t)]) # Friction term ψ.
        self.mst_sev = numpy.array([self._mst(x) for x in sol.t])# [Msun] Average stellar mass over time. 
        self.Np = numpy.array([self._N(x, y, z, u) for (x, y, z, u) in zip(self.M, self.fbh, self.mbh, sol.t)]) # Number of components. 
        self.mav = self.M / self.Np # [Msun] Average mass of cluster over time, includes BHs. No significant change is expected given that Nbh <= O(1e3), apart from the beginning where the difference is a few percent.
        self.Nbh = self.Mbh / self.mbh # Number of BHs.
        self.E = - self.r * self.G * self.M ** 2 / (4 * self.rh) # [pc^2 Msun / Myrs^2]
        self.xi = self._xi(self.rh, self.rt) # Evaporation rate.
        self.trh = numpy.array([self._trh(x, y, z, u, v) for x, y, z, u, v in zip(self.M, self.rh, self.fbh, self.mbh, self.t)]) # Relaxation within rh.
        self.trhstar = numpy.array([self._trhstar(x, y, z, u, v) for x, y, z, u, v in zip(self.M, self.rh, self.fbh, self.mbh, self.t)]) # Relaxation within rh.
        self.tcr = numpy.array([self._tcr(x, y) for x, y in zip(self.M, self.rh)]) # Crossing time.
        self.vesc = numpy.array([self._vesc(x, y) for x, y in zip(self.M, self.rh)]) # Escape velocity. 
        self.S = self.a11 * self.a12 ** (3 / 2) * (self.Mbh / self.Mst) ** self.b1 * (self.mbh / self.mst_sev) ** (3 / 2 * self.b2) # Parameter indicative of equipartition.
        
        # Check if we save results. The default option is to save the solutions of the differential equations as well as the tidal radius and average masses of the two components.
        if self.output:
           with open(self.outfile, "w") as f:
        # Header
              f.write("# t[Gyrs]    Mbh[msun]   Mst[msun]     rh[pc]     rt[pc]     mbh[msun]   mst[msun]\n")
        
        # Data rows
              for i in range(len(self.t)):
                  f.write("%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e\n" % (
                self.t[i], self.Mbh[i], self.Mst[i], self.rh[i], self.rt[i],
                self.mbh[i], self.mst_sev[i]
                ))

############################################################################################
"""
Notes:
- Another set of values that can be used is:
    zeta, beta, n, Rht, ntrh, b, Mval0, Mval_cc, S0 = 0.062, 0.077, 0.866, 0.046, 0.263, 2.2, 3.119, 3.855, 1.527
    zeta, beta, n, Rht, ntrh, b, Mval0, Mval_cc, S0 = 0.09, 0.058, 1.5, 0.074, 0.26, 2.14, 3.12, 3.12, 1.34. Computed for a fixed exponent n.

- An isolated cluster can be described with tidal=False. It can have stellar ejections or not.
- Simple cases where we have no stellar evolution can be described with nu=0.
- If we start with BHs in the balanced phase at t=0, use ntrh=0.
- If no change in the BH ejection rate is wanted, use a value for S0 quite close to 0 or disable running_bh_ejection_rate.
- A richer IMF can be specified by assigning values to additional mass breaks for the intervals, slopes and bins. Default option is 3. They must be inserted in sequence, so m_break5, a_slope4, nbin4 are the next inclusions. Masses must increase. 

Limitations:
- Dissolved clusters cannot be described. A varying mst is needed, and tidal effects must be considered in the stellar average mass.
- A more accurate connection between the unbalanced and the balanced phase may be needed. See EMACSS.
- Description of dark clusters has not been studied extensively. A different xi_bh as suggested in the comments may be more appropriate.
- Rotation, if activated, needs to be more accurate since now it remains constant.
- The special case of RV-filling clusters does not consider stellar induced mass loss rate. It can be inserted however in the odes function.
- Tidal shocks for interactions with GMC's for instance are not considered here.
"""
