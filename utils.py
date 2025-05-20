import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

#%%
def get_sectors():
    ### Sort sectors and countries in a way consistent with WIOT data
    return np.array(['AtB', 'C', '15t16', '17t18', '19', \
                     '20', '21t22', '23', '24', '25', '26',
                    '27t28', '29', '30t33', '34t35', '36t37', \
                    'E', 'F', 'WRT','H', '64', 'J','70', '71t74',\
                    'M', 'N', 'LOP'])

    
def get_imf_concordance():
    IIP_country_indices = np.array([10, 11, 18, 29, 26, 34, 40, 50, 51, 74, 52, 177, \
                                    62, 68, 69, 203, 77, 87, 90, 89, 93, 96, 98, 104, \
                                    115, 116, 109, 126, 122, 138, 155, 156, 158, 159, \
                                    171, 172, 184, 196, 187, 204])


    IMF_data = pd.read_excel('WEOOct2019all.xls')
    IMF_data.rename(columns = {"WEO Country Code" : "imf_code", 'ISO' : 'iso'}, inplace = True)
    IMF_to_iso = IMF_data[['imf_code', 'iso']].drop_duplicates().set_index('imf_code')
    iso_to_IMF = IMF_data[['imf_code', 'iso']].drop_duplicates().set_index('iso')
    
    return IMF_to_iso, iso_to_IMF

#%%
def get_countries():
    return np.array(['AUS', 'AUT', 'BEL', 'BGR', 'BRA', 'CAN', 'CHN', 'CYP', 'CZE',
       'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HUN',
       'IDN', 'IND', 'IRL', 'ITA', 'JPN', 'KOR', 'LTU', 'LUX', 'LVA',
       'MEX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'RUS', 'SVK', 'SVN',
       'SWE', 'TUR', 'TWN', 'USA', "RoW"]) ## NOTE: in theory 'RoW' should never be used


def get_country(country_index):
    return get_countries()[country_index]

def get_country_index(country_list):
    countries = get_countries()
    if np.ndim(country_list) == 0:
        return np.where(countries == country_list)[0][0]
    else:   
        return np.array([np.where(countries == country)[0][0] for country in country_list])

def get_eu_countries(): 
    
    eu_raw = ['AUT', 'BEL', 'BGR', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',\
                     'DEU', 'GRC', 'HUN', "HRV", 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', \
                     'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE']
    return [country for country in eu_raw if country in get_countries()]

def make_region_dict(raw_dict, EU = True):
    
    region_dict = raw_dict.copy()
    all_countries = get_countries()

    countries_in_regions = np.concatenate([region_dict[region] for region in region_dict])  

    # Check that all countries in region_dict are valid
    for country in countries_in_regions:
        if not np.isin(country, all_countries):
            raise ValueError(f"Country {country} in region_dict is not a valid country")
  
    # Find all countries that are not in a single region_dict     
    not_in_region = np.setdiff1d(all_countries, countries_in_regions)
    
    # If not_in_region is not empty, add it to the region_dict with the key 'ROW'
    if len(not_in_region) > 0:
        region_dict['ROW'] = not_in_region

    # Check each country appears exactly once in region_dict
    country_counts = {country: sum(country in region for region in region_dict.values()) 
                     for country in all_countries if country != "RoW"}
    
    for country, count in country_counts.items():
        if count != 1:
            raise ValueError(f"Country {country} appears {count} times in region_dict")

    return region_dict


def make_aggregator_matrix(region_dict):
    
    C_old = len(get_countries())
    C_new = len(region_dict)
    
    aggregator_matrix = np.zeros((C_new, C_old))
    
    for i, region in enumerate(region_dict):
        aggregator_matrix[i, get_country_index(region_dict[region])] = 1
    
    return aggregator_matrix



def make_region_series(region_dict):
    
    # Use the region_dict to create a pandas dataframe with columns 'country' and 'region'
    region_pd = pd.Series({country: region for region, countries in region_dict.items() for country in countries})
    
    return region_pd



def make_aggregator_matrix_weighted(region_dict, weights):
    # weights is a vector of length C_old
    C_old = len(get_countries())
    C_new = len(region_dict)
    
    aggregator_matrix = np.zeros((C_new, C_old))
    
    for i, region in enumerate(region_dict):
        for j in region_dict[region]:
            aggregator_matrix[i, get_country_index(j)] = weights[get_country_index(j)]
            
    # Normalize so row sums to 1
    aggregator_matrix = aggregator_matrix / np.sum(aggregator_matrix, axis=1)[:,None]
                     
    return aggregator_matrix


def make_keep_c_index(region_dict):
    
    all_countries = get_countries()
    
    # Check that all countries in region_dict are in all_countries
    for region in region_dict:
        if not np.isin(region_dict[region], all_countries):
            raise ValueError(f"Region {region} has countries not in all_countries")
    
    return np.array([get_country_index(region_dict[region]) for region in region_dict])

def make_collapse_matrix(region_dict):
    keep_c_index = make_keep_c_index(region_dict)
    return np.eye(len(keep_c_index))




def extend_capital(array, C, N, K, F):
    # Vector has length K. Extend to (C + N + K + F) with zeros in the new elements
    extended = np.zeros((C + N + K + F))
    extended[C+N:C+N+K] = array
    return extended



def leontief_inv(mat):
    """Computes the Leontief inverse (I - A)^(-1) for a given matrix A"""
    return np.linalg.inv(np.eye(len(mat)) - mat)


    

def integrate_derivative_irregular(d, dt = None):
    """
    Integrate f'(t) over an irregular grid of times by extrapolating the last derivative
    using actual time spacing.

    Parameters
    ----------
    d : 1D array of length N
        d[i] = f'(t_i), the derivative at time t_i.
    dt : 1D array of length N
        dt[i] is the time step from t_i to t_{i+1}.
        Thus, t_0 = 0, t_1 = dt[0], t_2 = dt[0] + dt[1], etc.

    Returns
    -------
    float
        The approximate integral from t_0=0 up to t_N = sum(dt).
        That is, f(t_N) - f(t_0).
    """
    
    T = d.shape[0]

    dt = dt if dt is not None else np.ones(T) / T
    
    if T < 2:
        raise ValueError("Need at least two derivative points to extrapolate.")

    # Build array of times t[i]
    t = np.zeros(T)
    for i in range(1, T):
        t[i] = t[i-1] + dt[i-1]

    # Final time is t_N = t[N-1] + dt[N-1]
    t_end = t[-1] + dt[-1]

    # Use the actual time difference to compute the slope
    # in the derivative (f'(t))
    delta_t = t[-1] - t[-2]
    slope = (d[-1] - d[-2]) / delta_t
    # Extrapolate derivative from t_{N-1} to t_{N}
    d_at_end = d[-1] + slope * (t_end - t[-1])


    # Extend arrays and do trapezoidal rule
    t_extended = np.append(t, t_end)

    if d.ndim == 1:
        d_extended = np.append(d, d_at_end)
    else:
        d_extended = np.vstack((d, d_at_end[None,:]))

    np.trapz(d_extended, x=t_extended, axis = 0)

    return np.trapz(d_extended, x=t_extended, axis = 0)


    #spline = CubicSpline(t_extended, d_extended, bc_type='natural')
    #val, _ = quad(spline, t_extended[0], t_extended[-1])
    #return val
    # return np.trapz(d_extended, x=t_extended)
    
    
    
