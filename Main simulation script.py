import numpy as np
import matplotlib.pyplot as plt
import time
from Development_Rate_and_Fecundity_Script import v_t
from Simulation_functions import TDMAsolver, generate_beta_pdf, pop_simulation_w_gen, cum_pop

# Actual Script
if __name__=='__main__':
    start_time = time.time()
    # Initializing Parameters:
    Nx = 200
    sigma = np.array([0.001, 0.00001])
    saveFig = True # If you want to save the simulation result figures
    saveFig_cum = True # If you want to save the cumulative plot figure
    saveFig_dir_cum = 'figures/cumulative plots/' # Change the directory as you want
    saveFig_dir_gen = 'figures/simulations/' # Change the directory as you want
    use_fecundity = False #Depends on if you want to use fecundity or no
    gen = 5 # number of generations considered
    init_cond = "[0.75, 1] uniform" # Other options are: "Uniform", 0, "[0.5, 1] uniform" or "[0, 0.5] uniform", "[0.75, 1] uniform" or 1, "[0.95, 1] uniform"
    in_num = 100 # Initial number of pests
    m = 0 # Death rate of pests

    # Choose True if you want to use a Beta distribution with given parameters below
    beta_ovipos_profile = True 
    bet_alpha = 3
    bet_beta = 9
    
    
    # Get the temp data from the .csv file
    file_path = 'Data/Generated 2023 Temperature Data.csv'
    
    # Pest development parameter values fitted to the Briere function.
    alpha = np.array([0.000067186, 0.000053013, 0.000055386])
    Tmin = np.array([0.00014821, 0.00005876, 0.00005775])
    Tmax = np.array([37.29, 35.98, 33.29])
    ad_a_s_w = np.array([0.0026, 4.2512])
    ad_b_s_w = np.array([19.2926, 232.5575])


    # Fecundity variables
    win_a, win_b, win_c = -0.054, 2.405, -18.708
    sum_a, sum_b, sum_c = -0.088, 4.282, -42.419

    stage_num = len(alpha)
    

# --------------------- Run the Simulation With the Generations --------------------- #
    dates, phi, N_t_c, w_end_dates = pop_simulation_w_gen(Nx, file_path, alpha, Tmin, Tmax, sigma,
                                        gen, saveFig, saveFig_dir_gen, init_cond, in_num, ad_a_s_w, ad_b_s_w,
                                                          win_a, win_b, win_c, sum_a, sum_b, sum_c, use_fecundity,
                                                          bet_alpha, bet_beta, beta_ovipos_profile, m)

    # Timing the simulation
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time for simulation: {execution_time:.2f} seconds")
    

# --------------------- Perform the Cumulative Entry Figures --------------------- #
    print('Now plotting cum sum plots')
    cum_pop(Nx, dates, sigma, N_t_c, saveFig_cum, saveFig_dir_cum, init_cond,
            in_num, w_end_dates, gen, beta_ovipos_profile, bet_alpha, bet_beta)


    # Displaying the figures on screen
    plt.show()






    
