In order to run the script first, expand the .zip folders, put them in the same location as the other .py files. Then just run the “Main simulation script.py” file. To change a parameter, only change the variables that are assigned under the “if __name__=='__main__':” part, which is also highlighted by the comment “Actual Script”.

The main script will perform the stochastic pesp population simulation via the functions defined in the “Simulation_functions.py”, v_t from the “Development_Rate_and_Fecundity_Script.py” file and generated temperature data from the Data folder. The results will appear on the screen in the form of plots. If you want to save the results automatically or not, adjust the boolean variables in the main script.

Please note that in order to not share the confidential information the actual data for the pest abundances and temperature are removed. The generated temperature data is generated randomly, only to show how the algorithm works.

I hope it will be useful.
