import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from Development_Rate_and_Fecundity_Script import v_t
import pandas as pd
from datetime import datetime
import math
from cycler import cycler
import sys
from scipy.stats import beta

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays

    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it,] = bc[it,] - mc*cc[it-1,] # bc = bc[npnewaxis, :] ve ac,cc = ac,cc[:, np.newaxis]
        dc[it,] = dc[it,] - mc*dc[it-1,]
        	    
    xc = bc
    xc[-1,] = dc[-1,]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
    return xc

# Function Generationg beta distribution with given parameters
def generate_beta_pdf(alpha, beta_param, N, loc=0, scale=1):
    """
    Generate the Beta PDF for N points between [0, 1], optionally transformed by location and scale.

    Parameters:
    - alpha (float): Shape parameter α > 0
    - beta_param (float): Shape parameter β > 0
    - N (int): Number of points in the interval
    - loc (float): Location parameter (default is 0)
    - scale (float): Scale parameter (default is 1)

    Returns:
    - x (np.ndarray): Points between [0, 1] transformed by loc and scale
    - y (np.ndarray): Beta PDF values at those points
    """
    # Generate N points in [0, 1] transformed by loc and scale
    x = np.linspace(0, 1, N)
    x_transformed = x * scale + loc
    # Compute Beta PDF
    y = beta.pdf(x, alpha, beta_param, loc=loc, scale=scale)
    return x_transformed, y

def pop_simulation_w_gen(Nx, file_path, alpha, Tmin, Tmax, sigma, gen,
                         saveFig, saveFig_dir, init, in_num, ad_a_s_w, ad_b_s_w,
                         a_win, b_win, c_win, a_sum, b_sum, c_sum, use_fecundity, bet_alpha, bet_beta, beta_ovipos_profile, m):

    """
    Simulates population dynamics of a multi-stage insect species across multiple generations
    using spatially explicit, temperature-dependent development and Gaussian dispersal.

    The model incorporates:
        - Temperature-driven development rates via a Brière function.
        - Multiple life stages (e.g., egg, neanid, nymph, adult).
        - Generational reproduction and fecundity.
        - Gaussian diffusion to simulate spatial spread.
        - Beta-distributed oviposition (optional).
        - Visualizations for each sigma value and generation.

    Parameters:
    -----------
    Nx : int
        Number of spatial discretization points.
    file_path : str
        Path to CSV file containing temperature data.
    alpha : float
        Scaling coefficient in the Brière development rate function.
    Tmin : float
        Minimum temperature for development.
    Tmax : float
        Maximum temperature for development.
    sigma : ndarray
        Array of standard deviations for Gaussian dispersal.
    gen : int
        Number of generations to simulate.
    saveFig : bool
        If True, plots will be saved to disk.
    saveFig_dir : str
        Directory path to save figures.
    init : str or int
        Initial distribution mode for the population.
        Options include "Uniform", "[0.5, 1] uniform", "[0.75, 1] uniform",
        "[0.95, 1] uniform", or positions 0 (x=0) and 1 (x=1).
    in_num : int
        Initial number of individuals.
    ad_a_s_w, ad_b_s_w : float
        Adult stage-specific parameters used in v_t (development rate) calculations.
    a_win, b_win, c_win : float
        Window parameters for adult stage (e.g., for photoperiod-based development).
    a_sum, b_sum, c_sum : float
        Parameters for cumulative effects in development (e.g., diapause).
    use_fecundity : bool
        Whether to use a time-varying fecundity function `b_t`.
    bet_alpha, bet_beta : float
        Alpha and beta parameters of the beta distribution for oviposition profile.
    beta_ovipos_profile : bool
        If True, beta distribution is used for spatial oviposition.
    m : float
        Mortality rate for all life stages.

    Returns:
    --------
    dates : list of datetime.datetime
        List of dates corresponding to the simulation time steps.
    phi : ndarray
        Full simulation output array with shape (stage, time, space, sigma, generation),
        representing the population density across time and space.
    N_t_c : ndarray
        Total population (integrated over space) by stage, time, sigma, and generation.
    w_end_dates : tuple of int
        Tuple containing (day, month) marking the end of diapause window.

    Notes:
    ------
    - The simulation uses finite difference methods and tridiagonal matrix solvers (TDMA).
    - Plotting occurs during the simulation. Results are optionally saved based on `saveFig`.
    - Each life stage is updated sequentially at each time step.
    - Fecundity is implemented at reproduction times using spatial oviposition profiles.

    """

    # Read the temperature data and evaluate development rate (v) as the briere function
    dates, v, w_end_dates, b_t = v_t(file_path, alpha, Tmin, Tmax, ad_a_s_w, ad_b_s_w,
                                     a_win, b_win, c_win, a_sum, b_sum, c_sum)
    
    stage_num = len(v[0,:]) # Get the number of stages
    v = v[:, :, np.newaxis] # Adjust the shape of development vector for vectorized operations below

    # Initialize the variables
    dx = 1/(Nx - 0.5)
    x = np.arange(dx/2,1,dx)
    sigma = sigma[:, np.newaxis, np.newaxis]
    dates = [d.to_pydatetime() for d in dates]
    Tfin = (dates[-1] - dates[0]).days
    Nt = len(dates)
    tim = np.linspace(0, int(Tfin), Nt)
    dt = (Tfin-1)/Nt
    s_dx2 = sigma/dx**2

    gen_arr = np.char.add('Gen: ',np.arange(gen).astype(str)) # will be used to label the generation info on plots

    # Initialize the solution variables
    phi = np.zeros(shape = (stage_num, Nt, Nx, len(sigma), gen))
    d = phi



    # ------------------ Setting the Initial Condition ------------------ #
    if init == "Uniform":
        phi[-1,0,:,:,0] = in_num/dx/Nx # uniform or at x=0 initial distribution
    elif init == 0:
        phi[-1,0,0,:,0] = in_num/dx # x=0 initial distribution
        init = "at 0"
    elif init == "[0.5, 1] uniform":
        phi[-1,0,Nx//2:,:,0] = in_num/dx/Nx*2 # last half uniform dist
    elif init == "[0, 0.5] uniform":
        phi[-1,0,:Nx//2,:,0] = in_num/dx/Nx*2 # first half uniform dist
    elif init == 1:
        phi[-1,0,-1,:,0] = in_num/dx # x=1 initial distribution
        init = "at 1"
    elif init == "[0.75, 1] uniform":
        phi[-1,0,Nx*3//4:,:,0] = in_num/dx/Nx*4 # last quarter uniform dist
    elif init == "[0.95, 1] uniform":
        phi[-1,0,int(Nx*9.5)//10:,:,0] = in_num/dx/Nx*20 # last one twentieth uniform dist
    
    # ------------------ Setting the Initial Condition ------------------ #




    # Calculations for all populations in the same for loop
    for t in tqdm(range(len(tim)-1), desc="Loading", miniters=int(Nt/10)): # The loading bar

        #############   Vector initialization for all populations   #############

        # Start initializing diagonal vector "b"
        b = np.zeros(shape=(Nx, len(sigma), len(v[t,:]), gen))
        b[1:,:,:,] = 1 + dt*(m + 2*s_dx2)
        b[0,:,:,:] = 1 + dt*(m + s_dx2 + v[t,:]/(2*dx))

        # Start initializing off diagonal vectors "a" and "c"
        a = np.zeros(shape=(Nx-1, len(sigma), len(v[t,:]), gen))
        a[0:-1, :,] = -dt*(v[t,:]/(2*dx) + s_dx2)
        a[-1,:,:,] = -2*dt*s_dx2 # My calculations

        c = np.zeros(shape=(Nx-1, len(sigma), len(v[t,:]), gen))
        c[0:,:,:,] = dt*(v[t,:]/(2*dx) - s_dx2) 

        # For fecundity, initialize b_t and f_x according to the inputs in the main code
        f_x = np.ones(Nx)
        if not use_fecundity:
            b_t = np.ones(len(b_t))

        if beta_ovipos_profile:
            x_grid, f_x = generate_beta_pdf(bet_alpha, bet_beta, Nx)
        
        # Perform first fecundity with the specified initial distribution in teh main code
        if init == "Uniform":
            d[0,t,0,:,1] = phi[0,t,0,:,1] + dt/dx * v[t,-1] * 2 * np.sum(phi[-1,t,:,:,0]*dx*b_t[t], axis=0)
        elif init == 0:
            d[0,t,0,:,1] = phi[0,t,0,:,1] + dt/dx * v[t,-1] * np.sum(phi[-1,t,:,:,0]*dx*b_t[t], axis=0)
        elif init == "[0.5, 1] uniform":
            d[0,t,0,:,1] = phi[0,t,0,:,1] + dt/dx * 4 * v[t,-1] * np.sum(phi[-1,t,:,:,0]*dx*b_t[t], axis=0)
        elif init == "[0.75, 1] uniform":
            d[0,t,0,:,1] = phi[0,t,0,:,1] + dt/dx * 8 * v[t,-1] * np.sum(phi[-1,t,:,:,0]*dx*b_t[t], axis=0)
        elif init == "[0.95, 1] uniform":
            d[0,t,0,:,1] = phi[0,t,0,:,1] + dt/dx * 40 * v[t,-1] * np.sum(phi[-1,t,:,:,0]*dx*b_t[t], axis=0)

        # The following reproduction of pests. 
        d[0,t,0,:,2:] = phi[0,t,0,:,2:] + dt/dx * v[t,-1] * np.sum(phi[-1,t,:,:,1:-1]
                                                                   *f_x[:,np.newaxis, np.newaxis]*dx*b_t[t], axis=0)   
        d[0,t,1:,:,:] = phi[0,t,1:,:,:]
        phi[0,t+1,:,] = TDMAsolver(a[:,:,0], b[:,:,0], c[:,:,0], d[0,t,])

        # For each stage, evaluate the solution
        for j in range(stage_num-1):
            i = j + 1
            d[i,t,0,:,] = phi[i,t,0,:,] + dt*(v[t,i-1]*phi[i-1,t, -1, :,])/dx
            d[i,t,1:,:,] = phi[i,t,1:,:,]
            phi[i,t+1,] = TDMAsolver(a[:,:,i], b[:,:,i], c[:,:,i], d[i,t,])

            

    #############   Plotting populations   #############
    N_t = np.sum(phi, axis=2)*dx
    stgs = ['Neanid Pop.', 'Nymph Pop.']

    # To plot each month on x-axis in dd.mm.yyyy format
    tick_num = 11 
    selected_ticks = []
    for t in range(tick_num):
        selected_ticks.append(dates[len(dates)*t//tick_num])
    formatted_ticks = [tick.strftime("%d/%b") for tick in selected_ticks]

    # Automatically adjusts the shape of the subplot
    ncols = math.ceil(math.sqrt(stage_num))
    nrows = math.ceil(stage_num/ncols)
    for i,s in enumerate(sigma[:,0,0]):
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(14, 7.5))
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(fr"Pupolation Dynamics, Gaussian Development, $\sigma=${s:.6f}, $\Delta$x={dx:.3f}")
        axs = axs.ravel()

        # Plotting Egg Simulation Results
        axs[0].plot(dates, N_t[0,:,i,], label = gen_arr)
        axs[0].set_title(r'Egg pop. vs $t$')        
        axs[0].set_xticks(selected_ticks)
        axs[0].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[0].set_ylabel(r'$N_{Egg}(t)$')

        #Plotting Neanid and Nymphs Simulation results
        for k in range(stage_num-2):
            l = k+1
            axs[l].plot(dates, N_t[l,:,i,], label = gen_arr)
            axs[l].set_xticks(selected_ticks)
            axs[l].set_xticklabels([d for d in formatted_ticks], rotation=90)
            axs[l].set_title(fr'{stgs[k]:s} vs $t$')
            axs[l].set_ylabel(r'$N_{%s}(t)$' %stgs[k])
        axs[2].legend()
        
        # Plotting Adult Simulation Results
        axs[-1].set_prop_cycle(cycler('color', plt.cm.tab10.colors))
        axs[-1].plot(dates, N_t[-1,:,i,], label=gen_arr)
        axs[-1].set_xticks(selected_ticks)
        axs[-1].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[-1].set_title(r'Adult pop. vs $t$')
        axs[-1].set_ylabel(r'$N_{Adult}(t)$')

        plt.subplots_adjust(hspace=0.7)
        
        #Saving the figure
        direc = saveFig_dir + f'With Generations, {in_num:d} eggs dist. {init:s}, diapause end on {w_end_dates[0]:d}.{w_end_dates[1]:d}, Simulation Results,sigma={s:6f}, Nx={Nx:d}.png'
        if saveFig:
            plt.savefig(direc)
            
    # N_t_c will be returned in the end
    N_t_c = N_t

    # Plot the results without the generation info
    N_t = np.sum(N_t, axis=-1) # Sum over all the generations
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(14, 7.5))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(fr"Pupolation Dynamics, Gaussian Development, $\Delta$x={dx:.3f}")
    axs = axs.ravel()
    colors = plt.cm.tab10(np.arange(stage_num))

    # For each sigma value
    for i,(s, color) in enumerate(zip(sigma[:,], colors)):

        # Plot the Egg Simulation Results
        axs[0].plot(dates, N_t[0,:,i], label=r'$\sigma$=%.6f' %s, color = color)
        axs[0].set_title(r'Egg pop. vs $t$')        
        axs[0].set_xticks(selected_ticks)
        axs[0].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[0].set_ylabel(r'$N_{Egg}(t)$')

        # Plot the Neanid Simulation Results
        axs[1].plot(dates, N_t[1,:,i,], label=r'$\sigma$=%.6f' %s, color = color)
        axs[1].set_xticks(selected_ticks)
        axs[1].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[1].set_title(r'Neanid Pop. vs $t$')
        axs[1].set_ylabel(r'$N_{Neanid}(t)$')

        # Plot the Nymph Simulation Results
        axs[2].plot(dates, N_t[2,:,i,], label=r'$\sigma$=%.6f' %s, color = color)
        axs[2].set_xticks(selected_ticks)
        axs[2].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[2].set_title(r'Nymph Pop. vs $t$')
        axs[2].set_ylabel(r'$N_{Nypmh}(t)$')
        axs[2].legend()

        # Plot the Adult Simulation Results
        axs[-1].plot(dates, N_t[-1,:,i,], label=r'$\sigma$=%.6f' %s, color = color)
        axs[-1].set_xticks(selected_ticks)
        axs[-1].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[-1].set_title(r'Adult pop. vs $t$')
        axs[-1].set_ylabel(r'$N_{Adult}(t)$')
    plt.subplots_adjust(hspace=0.7)
    
    #Saving the figure
    if saveFig:
            plt.savefig(saveFig_dir + f"Without Generation, {in_num:d} adults dist. {init:s}, Simulation Results, Nx={Nx:d}.png")
    return dates, phi, N_t_c, w_end_dates

def cum_pop(Nx, dates, sigma, N_t, saveFig, saveFig_dir_cum, init, in_num,
            w_end_dates, gen, beta_ovipos_profile, bet_alpha, bet_beta):
    """
    Plot and optionally save the cumulative population distribution across developmental stages over time.

    This function computes the cumulative distribution function (CDF) of the simulated populations across time,
    based on the total number of individuals per stage. It then generates plots for each stage and sigma value,
    visualizing how the cumulative fraction of individuals progresses over time.

    Parameters:
    -----------
    Nx : int
        Number of spatial discretization points.
    
    dates : list of datetime
        List of dates corresponding to each time step in the simulation.
    
    sigma : np.ndarray
        Array of sigma values (standard deviations of Gaussian development rates) to plot results for.
    
    N_t : np.ndarray
        4D array of population values with shape (stages, time steps, sigma values, generations).
    
    saveFig : bool
        Whether to save the generated figures to disk.
    
    saveFig_dir_cum : str
        Directory path where the cumulative population figures will be saved if `saveFig` is True.
    
    init : str or int
        String or integer indicating the initial spatial distribution of the population (e.g., "Uniform", 0, 1, etc.).
    
    in_num : int
        Number of initially distributed individuals.
    
    w_end_dates : tuple of int
        A tuple representing the end date of diapause (as day and month) for labeling saved figures.
    
    gen : int
        Number of generations simulated.
    
    beta_ovipos_profile : bool
        Whether a beta distribution was used for oviposition (reproduction) spatial profile.
    
    bet_alpha : int
        Alpha parameter of the beta distribution used in oviposition profile.
    
    bet_beta : int
        Beta parameter of the beta distribution used in oviposition profile.

    Returns:
    --------
    None
        The function creates and optionally saves cumulative population plots for each sigma value.
    """

    # To save the figure prettier
    if init == 0:
        init = "at 0"
    elif init ==1:
        init = 'at 1'

    # Check how many stages are present in the model
    stage_num = len(N_t[:,0,0])

    # Initialize variables
    dx = 1/(Nx - 0.5)
    gen_arr = np.char.add('Gen: ',np.arange(gen).astype(str))

    # To calculate the cumulative number of individuals
    sum_N_t = np.sum(N_t, axis=1)
    sum_N_t = sum_N_t[:,np.newaxis,]
    cdf = np.cumsum(N_t, axis=1)/sum_N_t
    
    # For plotting the dates
    days = np.arange(1,len(dates)+1)
    tick_num = 11
    selected_ticks = []
    formatted_ticks = []
    for t in range(tick_num):
        selected_ticks.append(dates[len(dates)*t//tick_num])
    formatted_ticks = [tick.strftime("%d/%b") for tick in selected_ticks]

    
    for i,s in enumerate(sigma):
        ncols = math.ceil(math.sqrt(stage_num))
        nrows = math.ceil(stage_num/ncols)
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(14, 7.5))
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(fr"Cum. Num. of Individuals""\n"
                     fr"Gaussian Development, $\sigma=${s:.6f}, $\Delta$x={dx:.3f}")
        axs = axs.ravel()
        
        # Plotting Egg Simulation Results
        axs[0].plot(dates, cdf[0,:,i,], label = gen_arr)
        axs[0].set_title(r'Egg pop. vs $t$')   
        axs[0].set_xticks(selected_ticks)
        axs[0].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[0].lines[0].remove()
        axs[0].set_ylabel(r'$N_{Egg}(t)$')

        # Plotting Neanid Simulation Results
        axs[1].plot(dates, cdf[1,:,i,], label=gen_arr)
        axs[1].set_xticks(selected_ticks)
        axs[1].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[1].set_title(r'Neanid Pop. vs $t$')
        axs[1].set_ylabel(r'$N_{Neanid}(t)$')
        
        # Plotting Nymph Simulation Results
        axs[2].plot(dates, cdf[2,:,i,], label=gen_arr)
        axs[2].set_xticks(selected_ticks)
        axs[2].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[2].set_title(r'Nymph Pop. vs $t$')
        axs[2].set_ylabel(r'$N_{Nypmh}(t)$')

            
        # Plotting Adult Sİmulation Results
        axs[-1].plot(dates, cdf[-1,:,i,], label=gen_arr)
        axs[-1].lines[0].remove()
        axs[-1].set_xticks(selected_ticks)
        axs[-1].set_xticklabels([d for d in formatted_ticks], rotation=90)
        axs[-1].set_title(r'Adult pop. vs $t$')
        axs[-1].set_ylabel(r'$N_{Adult}(t)$')        

        handles, labels = axs[1].get_legend_handles_labels()

        # Remove the first element
        axs[2].legend(handles[1:], labels[1:], loc='upper left', fontsize=8)
        
        # Adjust the spacing in the figure
        plt.subplots_adjust(hspace=0.5)
        

        #Saving the figure
        direc = ''
        if beta_ovipos_profile:
            direc = saveFig_dir_cum + f'ovipos. prof. beta({bet_alpha:d}, {bet_beta:d}), '
            
        direc += f'sigma = {s:.6f}, diapause end on {w_end_dates[0]:d}.{w_end_dates[1]:d}, with {in_num:d} adults distributed {init:s}.png'
        if saveFig:
            plt.savefig(direc)
    return
