import numpy as np
import pandas as pd

def briere_function(T, a, Tmin, Tmax):
    """
    Brière function with b = 0.5.
    
    Parameters:
    - T: Array of temperatures (numpy array or list).
    - a: Scaling constant.
    - Tmin: Minimum temperature threshold.
    - Tmax: Maximum temperature threshold.
    
    Returns:
    - Response (R) for each temperature in T.
    """
    return np.where((T > Tmin) & (T < Tmax), T * (T - Tmin) * np.sqrt(Tmax - T) * a, 0)

def fecundity(T, a, b, c):
    """
    Creates a concave parabola according to temperature values as
    fec = a*T^2 + b*T + c.
    Pay attention the script only yields the positive part of the parabola.
    """
    res = a*T**2 + b*T + c
    return np.where(res<0, 0, res)

def adult_dev(T, a, b):
    """
    The function for fitting the adult development using the so called Erying rate function.

    Parameters:
    - T (float): The temperature value.
    - a (float): Multiplicative parameter for the development rate.
    - b (float): Exponent parameter for the development rate.

    Returns:
    -res (np.array): The development rate based on the Erying rate function.
    """
    res = a * np.abs(T)*np.exp(-b/np.abs(T))
    return res.flatten()

def v_t(file_path, a, Tmin, Tmax, ad_a_s_w, ad_b_s_w,
        a_win, b_win, c_win, a_sum, b_sum, c_sum):
    """
    Calculates temperature-dependent development and fecundity rates for a population model
    using temperature data from an Excel sheet, and distinguishes between summer and winter periods.

    Parameters:
    ----------
    file_path : str
        Path to the .csv file containing input data.
    a : array-like
        Coefficients for the Brière function used to model development rates.
    Tmin : float
        Minimum temperature threshold for development.
    Tmax : float
        Maximum temperature threshold for development.
    ad_a_s_w : tuple
        Coefficients (summer_a, winter_a) for adult development rate function.
    ad_b_s_w : tuple
        Coefficients (summer_b, winter_b) for adult development rate function.
    a_win, b_win, c_win : float
        Coefficients for the winter fecundity function.
    a_sum, b_sum, c_sum : float
        Coefficients for the summer fecundity function.

    Returns:
    -------
    dates : list of datetime
        List of hourly timestamps from the Excel sheet.
    v : ndarray
        Array of shape (n_hours, len(a)+1) containing temperature-dependent development rates.
        The last column represents adult development, adjusted seasonally.
    w_end_dates : list
        List representing the end of the winter period as [month, day].
    b_t : ndarray
        Array of temperature-dependent fecundity rates, adjusted seasonally and halved
        to account for female-only reproduction.

    Notes:
    -----
    - The function uses the Brière model for developmental rates and a temperature-dependent
      function for fecundity.
    - The adult development and fecundity rates are adjusted based on whether the period falls 
      in summer or winter.
    - Winter is defined from October 1 to January 25 (inclusive), outside of which it is considered summer.
    """
    
    data = pd.read_csv(file_path, delimiter=';')

    # Extract columns as arrays
    dates = data['Hourly Dates'].to_list()
    dates = pd.to_datetime(dates, dayfirst=True)
    days = data['Days']
    days = pd.to_datetime(days, dayfirst=True)
    Tem = pd.to_numeric(data['Temperatures'], errors='coerce')
    Temp = Tem.to_numpy()[:, np.newaxis]
    
    v = np.zeros(shape=(len(Temp), len(a) + 1))

    #Performing fecundity calculations
    b_t_winter = fecundity(Tem, a_win, b_win, c_win)
    b_t_sum = fecundity(Tem, a_sum, b_sum, c_sum)
    b_t = np.zeros(len(b_t_winter))

    # Choose when the winter starts and ends
    w_end_dates=[1,25]
    win_start_ind = days[(days.dt.month == 10) & (days.dt.day == 1)].index.tolist()[0]
    win_end_ind = days[(days.dt.month == w_end_dates[0]) & (days.dt.day == w_end_dates[1])].index.tolist()[0]
    
    v[:, :-1] = briere_function(Temp, a, Tmin, Tmax)

    # Adjust the adult development rate (v) according to adults winter or summer form
    v[0:win_end_ind, -1] = adult_dev(Temp[0:win_end_ind], ad_a_s_w[1], ad_b_s_w[1])
    v[win_end_ind:win_start_ind, -1] = adult_dev(Temp[win_end_ind:win_start_ind], ad_a_s_w[0], ad_b_s_w[0])
    v[win_start_ind:, -1] = adult_dev(Temp[win_start_ind:], ad_a_s_w[1], ad_b_s_w[1]).ravel()

    # Adjust the fecundity rate function b_t according to adults winter or summer form
    b_t[0:win_end_ind] = b_t_winter[0:win_end_ind]
    b_t[win_end_ind: win_start_ind] = b_t_sum[win_end_ind: win_start_ind]
    b_t[win_start_ind: -1] = b_t_winter[win_start_ind: -1]
    b_t = b_t/2 # only females lay eggs
    
    return dates, v, w_end_dates, b_t
