'''
Re-calibrate parameter values given the case (eg. Benchmark, Cobb-Douglas, no markup) we consider.
Author: Yutong Zhong
'''
import numpy as np
import pandas as pd
from scipy import optimize as opt
from einops import rearrange
import pickle
import utils
from numba import njit
import utils 

import scipy.sparse

#import Multi_country_calibration as MC
#from Multi_country_calibration import *


def calibrate_BGP(
        region_dict, \
        closed_economy = None, \
        closed_country_index = None, \
        theta = 5, \
        KORV = False, \
        markups = True, \
        time_smoothing = True , \
        static = False, \
        full_effect = True, 
        dual = False,
        modern_share = None,
        dual_cap_ratio = None ,
        dΩ_tilde_dp = True,
        name = "",
        capital_labor = None):
    
    print("Calibrate " + name)
    if static:
        return calibrate_static(region_dict, closed_economy, closed_country_index, KORV, time_smoothing, full_effect, theta)
   
    parameters, _ =  make_BGP_objects(region_dict, closed_economy, closed_country_index, time_smoothing)

    C = parameters['C'] # number of countries
    F_c = 3
    N_c = 26 
    K_c = N_c
    F = F_c * C
    N = N_c * C
    K = K_c * C
    

    if dual:
        N_c *=2 
        K_c *=2
        N *= 2
        K *= 2
        
        K_c_raw = K_c // 2
        N_c_raw = N_c // 2
        N_raw   = N // 2
        K_raw   = K // 2


    ρ_c = np.array(parameters['rho_c']) # discount factor (shape: C x 1)
    ν_c = np.array(parameters['nu_c']) # mortality rate (shape: C x 1)
    γ = parameters['gamma'] # intertemporal elasticity of substitution (scalar)
    g_A = parameters['g_A'] # technological improvement (scalar)
    g_L = parameters['g_L']# population growth (scalar)
    g = g_A + g_L
    θ = theta * np.ones(C+N+K+F) # law of elasticity  (shape: C+N+K+F x 1)

    δ = np.array(parameters['delta']) if not dual else np.concatenate([parameters['delta'], parameters['delta']]) # depreciation rate (shape: K x 1)
    # Ω_tilde = parameters['Omega_tilde'] # cost-based IO (shape: C+N+K+F x C+N+K+F)
    ### define markups μ and μ_tilde

    if dual:
        C_idx_base = slice(0, C)
        N_idx_base = slice(C, C + N_raw)
        K_idx_base = slice(C + N_raw, C + N_raw + K_raw)
        F_idx_base = slice(C + N_raw + K_raw, C + N_raw + K_raw + F)

    μ_raw = np.array(parameters['mu']) 

    μ = μ_raw if not dual else np.concatenate([np.ones(C), np.tile(μ_raw[N_idx_base], 2), np.tile(μ_raw[K_idx_base],2),μ_raw[F_idx_base] ])
    μ_tilde = np.ones(N)

    ### define tariff t
    t = np.zeros((C+N+K+F, C+N+K+F))


    r_i = np.array(parameters['r_i']) if not dual else np.tile(parameters['r_i'], 2) # risky rate of assets (shape: K x 1)
    r = parameters['r'] # risk free rate (scalar)
    σ = np.array(parameters['sigma']) if not dual else np.tile(parameters['sigma'], 2) # asset risk (shape: K x 1)
    ψ = np.array(parameters['psi']) if not dual else np.tile(parameters['psi'], 2) # financial friction (shape: K x 1)
    S_c = np.array(parameters['S_c']) # Sharpe ratio (shape: C x 1)



    Ω_tilde = np.array(parameters['Omega_tilde'])


    Φ_raw   = np.array(parameters['Phi'])
    Φ = Φ_raw if not dual else np.concatenate([Φ_raw[:C], np.zeros(N + K + F)])

    Φ_c = Φ[:C]



    N_to_C = np.zeros((C + N + K + F, C))
    for n in range(C):
        N_to_C[n, n] = 1
        if not dual:
            N_to_C[ C + n * N_c : C + (n+1) * N_c, n] = 1
            N_to_C[C + N + n * K_c : C + N + (n+1) * K_c, n] = 1
        else:
            N_to_C[C + n * N_c // 2: C + (n+1) * N_c //2, n] = 1
            N_to_C[C + N //2  + n * N_c // 2 :  C + N //2  + (n + 1) * N_c // 2, n] = 1
            N_to_C[C + N + n * K_c // 2 : C + N + (n+1) * K_c // 2, n] = 1
            N_to_C[C + N + K//2 + n * K_c // 2 : C + N + K//2 + (n+1) * K_c // 2, n] = 1
        
        
        N_to_C[C + N + K + n * F_c : C + N + K + (n+1) * F_c, n] = 1


    if dual:
        
        modern_share_vec = np.repeat(modern_share, N_c // 2)
        
        # Define dual sector parameters
        # Create temporary variables for original matrix sections
        Omega_CN = Ω_tilde[C_idx_base, N_idx_base]  # Original C,N block 
        Omega_NN = Ω_tilde[N_idx_base, N_idx_base] # Original N,N block
        Omega_KN = Ω_tilde[K_idx_base, N_idx_base]  # Original K,N block
        Omega_NK = Ω_tilde[N_idx_base, K_idx_base] # Original N,K block
        Omega_NF = Ω_tilde[N_idx_base, F_idx_base]  # Original N,F block
        # Calculate capital intensity ratios based on equations
        denominator = 1 - modern_share_vec + dual_cap_ratio * modern_share_vec
        
        
        Omega_N1K1 = Omega_NK @ np.diag(dual_cap_ratio / denominator) # First sector
        Omega_N2K2 = Omega_NK @ np.diag(1 / denominator) # First sector
        
        
        idx_C  = slice(0, C)
        idx_N1 = slice(C, C + N_raw)
        idx_N2 = slice(C + N_raw, C + N)
        idx_K1 = slice(C + N , C  + N +  K_raw)
        idx_K2 = slice(C + N + K_raw , C  + N + K)
        idx_F  = slice(C + N + K, C + N + K + F)
        
        
        Ω_tilde_new = np.zeros((C+N+K+F, C+N+K+F))
        

        Ω_tilde_new[idx_C, idx_N1]  = Omega_CN @ np.diag(modern_share_vec)
        Ω_tilde_new[idx_C, idx_N2]  = Omega_CN @ np.diag(1 - modern_share_vec)
        Ω_tilde_new[idx_N1, idx_N1] = Omega_NN @ np.diag(modern_share_vec)
        Ω_tilde_new[idx_N1, idx_N2] = Omega_NN @ np.diag(1 - modern_share_vec)
        
        Ω_tilde_new[idx_N2, idx_N1]  = Omega_NN @ np.diag(modern_share_vec)
        Ω_tilde_new[idx_N2, idx_N2]  = Omega_NN @ np.diag(1 - modern_share_vec)

        Ω_tilde_new[idx_N1, idx_K1]  = Omega_N1K1
        Ω_tilde_new[idx_N2, idx_K2]  = Omega_N2K2

        Ω_tilde_new[idx_K1, idx_N1]  = Omega_KN @ np.diag(modern_share_vec)
        Ω_tilde_new[idx_K1, idx_N2]  = Omega_KN @ np.diag(1 - modern_share_vec)

        Ω_tilde_new[idx_K2, idx_N1]  = Omega_KN @ np.diag(modern_share_vec)
        Ω_tilde_new[idx_K2, idx_N2]  = Omega_KN @ np.diag(1 - modern_share_vec)

        Omega_NF_norm     = Omega_NF  / np.sum(Omega_NF, axis = 1)[:, np.newaxis]
        labor_shares      = 1 - np.sum(Ω_tilde_new, axis = 1, keepdims=True)
                
        Ω_tilde_new[idx_N1, idx_F] = labor_shares[idx_N1]  * Omega_NF_norm   
        Ω_tilde_new[idx_N2, idx_F] = labor_shares[idx_N2]  * Omega_NF_norm   

        Ω_tilde = Ω_tilde_new



    τ_c = 0 * np.ones(C)  # initial capital taxes ## NOTE: require data (shape: C x 1)
    selected_countries = region_dict.keys()
    print('Countries:', list(selected_countries))

    '''
    changes to make for μ = 1 for capital goods
    '''
    if not markups:
        δ = 1000000000 * np.ones(K) # extremely large depreciation rate 
        # μ[C+N:C+N+K] = (r_i + δ)/(g + δ)
        μ[C+N:C+N+K] = np.ones(K)

    Ω       = Ω_tilde / μ[:,None]
    Ψ       = np.linalg.inv(np.eye(C+N+K+F) - Ω)
    Ψ_tilde = np.linalg.inv(np.eye(C+N+K+F) - Ω_tilde)
    λ       = Φ @ Ψ
    λ_tilde = Φ @ Ψ_tilde


    ### Define more equations
    g_ωc = γ * ((r - ρ_c) + ((γ + 1) / 2) * (S_c ** 2)) # (shape: C x 1)
    χ_c = (ν_c + g_L) / (ν_c + g - g_ωc) # (shape: C x 1)
    D_c = np.zeros(C)
    b_c = parameters['b_c'].flatten()
    T_c = (r-g) * b_c
    S_c_tilde = np.sum(np.reshape(λ[C+N+K:], (C, F_c)) * (1+D_c)[:,None], axis=1) / ((1-τ_c) * r + ν_c - g_A) * (χ_c - 1)

    if not dual:
        
        λ_GNE = λ[:C] + np.sum(np.reshape(λ[C+N:C+N+K]/μ[C+N:C+N+K], (C, K_c)), axis=1)
        
    else:
        λ_GNE = λ[:C] + np.sum(np.reshape(λ[idx_K1]/μ[idx_K1] + λ[idx_K2]/μ[idx_K2], (C, K_c_raw)), axis=1)

    print('λ_GNE by country is: ',  λ_GNE)



    #%% Interactive Window Break Point


    λ_sum_cap   = np.sum(np.reshape(λ[C+N:C+N+K]/(r_i+δ),(C,N_c)),axis=1)
    λ_sum       = np.sum(np.reshape(λ[C+N:C+N+K],(C,N_c)),axis=1)
    weights_cap = np.reshape(λ[C+N:C+N+K]/(r_i+δ),(C,N_c))/np.reshape(λ_sum_cap,(-1,1))      
    weights     = np.reshape(λ[C+N:C+N+K],(C,N_c))/np.reshape(λ_sum,(-1,1))
    r_bar       = np.sum(weights_cap*np.reshape(r_i,(C,N_c)),axis=1)
    mu_bar      = 1/np.sum(weights*np.reshape(1/μ[C+N:C+N+K],(C,N_c)),axis=1)
    print('Average wedge on capital', 1/np.sum(weights*np.reshape(1/μ[C+N:C+N+K],(C,N_c)),axis=1))
    print('Return to capital', np.sum(weights_cap*np.reshape(r_i,(C,N_c)),axis=1))


    # Solve for Psi's to be use in equilibrium conditions
    # N x C matrix that picks out the country associated with each element n \in N 
    # Padding factor with zeros to match the shape of Ψ

    
    
    


    a_eq1      = np.diag(1 - 1/μ)
    a_eq3      = np.diag(utils.extend_capital(r_i/(r_i + δ), C, N, K, F))    
    a_eq4      = np.diag(utils.extend_capital((σ * ψ / (r_i + δ)), C, N, K, F))
    a_eq5      = np.diag(utils.extend_capital(1 / (r_i + δ), C, N, K, F))     
    
    a_eq6      = np.eye(C + N + K + F)
    a_eq6      = a_eq6[:, C + N + K:]
    
    a_list     = [a_eq1, a_eq3, a_eq4, a_eq5]

    Psi_mats    = [Ψ @ a @ N_to_C for a in a_list]
    Psi_mats.append(Ψ @ a_eq6)

    mu_mats    =  [(Ψ - np.eye(len(Ψ))) @  a @ N_to_C for a in a_list]
    mu_mats.append((Ψ - np.eye(len(Ψ))) @ a_eq6)

    t_mats      = [Ψ @ a @ N_to_C for a in a_list]
    t_mats.append(Ψ  @ a_eq6)

    eq_matrices = \
        {"Psi_mats" : Psi_mats, "mu_mats" : mu_mats, "t_mats" : t_mats}
    
    # dictionary of steady state values
    

    ss = {'ρ_c': ρ_c, 'σ': σ, 'δ': δ, 'ν_c': ν_c, 'ψ': ψ, 'γ': γ, \
                'g_A': g_A, 'g_L': g_L, 'g': g, 'θ': θ, 't': t, 'τ_c': τ_c, \
                'μ_tilde': μ_tilde, 'λ': λ, 'λ_tilde': λ_tilde, 'λ_GNE': λ_GNE, 'Ω': Ω, 'Ω_tilde': Ω_tilde, \
                'Ψ': Ψ, 'Ψ_tilde': Ψ_tilde, 'Φ': Φ, 'Φ_c':Φ_c, 'r': r, 'r_i': r_i, \
                'S_c': S_c, 'S_c_tilde': S_c_tilde, 'g_ωc': g_ωc, 'μ': μ, \
                'χ_c': χ_c, 'D_c': D_c, 'b_c': b_c, 'T_c':T_c, 'C': C, 'N':N, 'K':K, \
                'F':F, 'N_c':N_c, 'K_c':K_c, 'F_c':F_c, 'KORV': KORV,  'countries' : list(region_dict.keys()),  \
                'keep_c_index' : region_dict, 'eq_matrices': eq_matrices, 'full_effect': full_effect, 'static': static,\
                'r_bar': r_bar, 'mu_bar': mu_bar, 'dual' : 1 if dual else 0,
                'C_agg' : N_to_C.T,
                'C_map' : np.argmax(N_to_C, axis=1)}
    
    print("Calculate dΩ_tilde_dp...")
    if dΩ_tilde_dp:
        C_map = ss['C_map']
        if capital_labor is None:
            dΩ_tilde_dp  =  make_dOmega_tilde_dp_fast(C, N, K, F, Ω_tilde, N_c, K_c, F_c, θ, C_map)
            ss['dΩ_tilde_dp'] = dΩ_tilde_dp
        else:
            print("Did we get here?")
            dΩ_tilde_dp  =  make_dOmega_tilde_dp_capital_labor(ss, capital_labor)
            ss['dΩ_tilde_dp'] = dΩ_tilde_dp
            
        
        # Reshape dΩ_tilde_dp to combine first two dimensions and store as sparse matrix
        shape = dΩ_tilde_dp.shape
        reshaped = dΩ_tilde_dp.reshape(shape[0]*shape[1], shape[2])
        ss['dΩ_tilde_dp_sparse'] = scipy.sparse.csr_matrix(reshaped)
        
    
    print("Done!")
    print("")

    return ss


#%%
def calibrate_static(\
        region_dict, \
        closed_economy = None, \
        closed_country_index = None, \
        KORV = False, \
        time_smoothing = True, \
        full_effect = True,
        theta = None):
    
    path = "@Import"
    parameters, raw_data = make_BGP_objects(region_dict, closed_economy, closed_country_index, time_smoothing)
    bigY = raw_data['bigY'] 
    C = raw_data['C'] 
    F_c = 4
    F = F_c * C
    N_c = 27
    N = N_c * C
    years = np.arange(1995, 2010)
    if time_smoothing == False:
        years = np.array([1997])
        bigY = np.reshape(bigY, (1, C+N+N+C*3, C+N+N+C*3))
    new_bigY = construct_static_big_Y(bigY, len(years), C, N, N_c, F, F_c)
    Φ_c = np.mean(np.sum(new_bigY[:,:C,:], axis=2) / np.sum(new_bigY[:,:C,:], axis=(1,2))[:,None], axis=0)
    education_index = 24
    education_indices = np.arange(C+education_index, C+N, N_c)
    bigY_no_education = np.delete(new_bigY, education_indices, axis=1) 
    bigY_no_education = np.delete(bigY_no_education, education_indices, axis=2) 

    #### Compute the primitives
    Ω_tilde = np.mean(construct_big_omega_tilde_F(F, bigY_no_education), axis=0) ## shape: C+N+F x C+N+F  

    N_c = 26
    N = N_c * C
    Φ = np.zeros(C+N+F)
    # Φ_c = parameters['Phi_static']
    Φ[:C] = Φ_c.copy()
    μ = np.ones(C+N+F)
    Ω = Ω_tilde / μ[:,None]
    Ψ = np.linalg.inv(np.eye(C+N+F) - Ω)
    Ψ_tilde = np.linalg.inv(np.eye(C+N+F) - Ω_tilde)
    λ = Φ @ Ψ # shape: C+N+F x 1
    λ_tilde = Φ @ Ψ_tilde # shape: C+N+F x 1
    #### Other parameter values
    θ = 5 * np.ones(C+N+F) if theta is None else theta * np.ones(C+N+F) # law of elasticity  (shape: C+N+F x 1)
    μ_tilde = np.ones(N) # depending on the goods tax/subsidy
    D_c = np.zeros(C)
    t = np.zeros(C+N+F)

    selected_countries = list(region_dict.keys())
    ss = {}
    
    # Pad it with empty stuff
    
    # Get parameters from parameters dict
    S_c = np.array(parameters['S_c']) # Sharpe ratio (shape: C x 1)
    r = parameters['r'] # risk free rate (scalar)
    ρ_c = np.array(parameters['rho_c']) # discount factor (shape: C x 1)
    ν_c = np.array(parameters['nu_c']) # mortality rate (shape: C x 1)
    γ = parameters['gamma'] # intertemporal elasticity of substitution (scalar)
    g_A = parameters['g_A'] # technological improvement (scalar)
    g_L = parameters['g_L']# population growth (scalar)
    g = g_A + g_L

    g_ωc = γ * ((r - ρ_c) + ((γ + 1) / 2) * (S_c ** 2)) # (shape: C x 1)
    χ_c = (ν_c + g_L) / (ν_c + g - g_ωc) # (shape: C x 1)
    D_c = np.zeros(C)
    # Populate ss with parameters and initial values
    ss.update({
        'K': 0,
        'r': r,
        'g': g,
        'b_c': np.zeros(C),
        'g_ωc': g_ωc,
        'χ_c': χ_c,
        'r_i': np.zeros(0),
        'S_c': S_c,
        'S_c_tilde':S_c,
        'ν_c': ν_c,
        'g_A': g_A,
        'g_L': g_L,
        'τ_c': np.zeros(C),
        'δ': np.zeros(0),
        'σ': np.zeros(0),
        'ψ': np.zeros(0),
        'γ': γ ,
        'K_c': 0
    })

    
    N_to_C = np.zeros((C + N + F, C))
    for n in range(C):
        N_to_C[n, n] = 1
        N_to_C[ C + n * N_c : C + (n+1) * N_c, n] = 1
        
        N_to_C[C + N + n * F_c : C + N + (n+1) * F_c, n] = 1

    a_eq1      = np.diag(1 - 1/μ)
    a_eq3      = np.diag(utils.extend_capital(np.zeros(0), C, N, 0, F))    
    a_eq4      = np.diag(utils.extend_capital(np.zeros(0), C, N, 0, F))
    a_eq5      = np.diag(utils.extend_capital(np.zeros(0), C, N, 0, F))     
    
    a_eq6      = np.eye(C + N + F)
    a_eq6      = a_eq6[:, C + N:]
    
    a_list     = [a_eq1, a_eq3, a_eq4, a_eq5]

    Psi_mats    = [Ψ @ a @ N_to_C for a in a_list]
    Psi_mats.append(Ψ @ a_eq6)

    mu_mats    =  [(Ψ - np.eye(len(Ψ))) @  a @ N_to_C for a in a_list]
    mu_mats.append((Ψ - np.eye(len(Ψ))) @ a_eq6)

    t_mats      = [Ψ @ a @ N_to_C for a in a_list]
    t_mats.append(Ψ  @ a_eq6)

    eq_matrices = \
        {"Psi_mats" : Psi_mats, "mu_mats" : mu_mats, "t_mats" : t_mats}
        
        
    ss.update({
        'θ': θ, 'μ_tilde': μ_tilde, 'λ': λ, 'λ_tilde': λ_tilde, 'Ω': Ω, 'Ω_tilde': Ω_tilde,
        'Ψ': Ψ, 'Ψ_tilde': Ψ_tilde, 'Φ': Φ, 'μ': μ, 'D_c': D_c, 'C': C, 'Φ_c' : Φ_c, 
        'N': N, 'F': F, 'N_c': N_c, 'F_c': F_c, 'KORV': KORV, 't': t,
        'countries': selected_countries, 'keep_c_index': region_dict, 'static': True, 'full_effect': full_effect,
        'C_agg' : N_to_C.T,
        'C_map': np.argmax(N_to_C, axis=1), 'eq_matrices': eq_matrices
    })
    
    
    return ss




###################################################################
# functions used by static model 
def construct_static_big_Y(bigY_no_education, years, C, N, N_c, F, F_c):
    new_bigY = np.zeros((years, C+N+F, C+N+F))
    for c in range(C):
        #### capital endowment (add to consumption)
        # new_bigY[:, c, C+N::F_c] = Consumption_K[c,:,:].T
        new_bigY[:, c, C:C+N] += np.sum(bigY_no_education[:, C+N+c*N_c:C+N+(c+1)*N_c, C:C+N], axis=1)
        #### regular goods consumption
        new_bigY[:, c, C:C+N] += bigY_no_education[:, c, C:C+N]
        #### regular goods usage of regular goods
        new_bigY[:, C:C+N, C:C+N] = bigY_no_education[:, C:C+N, C:C+N]
        #### regular goods usage of capital goods
        new_bigY[:, C+c*N_c:C+(c+1)*N_c, C+N+c*F_c] = np.sum(bigY_no_education[:, C:C+N, C+N:C+2*N], axis=2)[:,c*N_c:(c+1)*N_c]
        #### regular goods usage of primary factors (high-skill, medium-skill and low-skill)
        new_bigY[:, C+c*N_c:C+(c+1)*N_c, C+N+c*F_c+1] = np.sum(bigY_no_education[:, C:C+N, C+2*N::3], axis=2)[:,c*N_c:(c+1)*N_c]
        new_bigY[:, C+c*N_c:C+(c+1)*N_c, C+N+c*F_c+2] = np.sum(bigY_no_education[:, C:C+N, C+2*N+1::3], axis=2)[:,c*N_c:(c+1)*N_c]
        new_bigY[:, C+c*N_c:C+(c+1)*N_c, C+N+c*F_c+3] = np.sum(bigY_no_education[:, C:C+N, C+2*N+2::3], axis=2)[:,c*N_c:(c+1)*N_c]
    return new_bigY





def construct_big_omega_tilde_F(F, bigY):
    shape = bigY.shape
    Omega_tilde = np.zeros((shape[0], shape[1], shape[2]))
    for i in range(shape[1] - F):
        Omega_tilde[:, i, :] = np.divide(bigY[:, i, :], np.sum(bigY[:, i, :], axis=1)[:,None], out=np.zeros_like(bigY[:, i, :]), where=np.sum(bigY[:, i, :], axis=1)[:,None]!=0)

    return Omega_tilde


@njit
def make_dOmega_tilde_dp(ss):
    C, N, K, F, Ω_tilde, N_c, K_c, F_c = ss['C'], ss['N'], ss['K'], ss['F'], ss['Ω_tilde'], ss['N_c'], ss['K_c'], ss['F_c']
    Nfull = C + N + K + F
    N_nest = 1 + N_c + K_c + F_c
    #N_nest = 1 + N_c + 1

    θ = np.array([1] + [ss['θ'][0]]*N_c + [1]*(K_c + F_c))
    #θ = np.array([1] + [ss['θ'][0]]*N_c + [capital_labor])
    

    ind2nest = np.zeros((Nfull, N_nest))
    
    ind2nest[0:3,0] = 1
    

    for n in range(N_c):
        ind2nest[slice(C + n, C + N , N_c), 1 + n] = 1

    for k in range(K_c):
        ind2nest[slice(C + N + k, C + N + K, K_c), 1 + N_c + k] = 1

    for f in range(F_c):
        ind2nest[slice(C + N + K + f, C + N + K + F, F_c), 1 + N_c + K_c + f] = 1

    #ind2nest[slice(C + N, C + N + K + F), -1] = 1
    
    # Create mapping from (nest,c) to j indices
    nest_c_to_j = np.zeros((N_nest, C), dtype=int)
    for j in range(Nfull):
        for nest in range(N_nest):
            for c in range(C):
                if ind2nest[j,nest] == 1 and ss['C_map'][j] == c:
                    nest_c_to_j[nest,c] = j

    # Construct Omega matrix mapped to nest,c dimensions
    ind2nestC = np.zeros((Nfull, N_nest, C))
    for i in range(Nfull):
        for nest in range(N_nest):
            for c in range(C):
                j = nest_c_to_j[nest,c]
                ind2nestC[i,nest,c] = Ω_tilde[i,j]

    # Sum across countries to get Ω̃_inest            
    Ω̃_inest = np.sum(ind2nestC, axis=2)

    # Create W array with normalized weights
    W = np.zeros((Nfull, N_nest, C))
    country_sums = np.sum(ind2nestC, axis=2, keepdims=True)
    nonzero_mask = (country_sums != 0)
    W = np.where(nonzero_mask, ind2nestC / country_sums, 0)

    # Create the derivative of W with respect to prices
    C = W.shape[2]
    dW = np.zeros((W.shape[0], W.shape[1], C, C))

    for i in range(W.shape[0]):    # Loop over goods
        for n in range(W.shape[1]): # Loop over nests
            for c1 in range(C):     # First country index
                for c2 in range(C): # Second country index
                    if c1 == c2:
                        # Diagonal term
                        dW[i,n,c1,c2] = W[i,n,c1]
                    # Subtract outer product term
                    dW[i,n,c1,c2] -= W[i,n,c1] * W[i,n,c2]
    # Multiply by (1-θ) factor                
    dW *= (1-θ[np.newaxis, :, np.newaxis, np.newaxis])

    dW_full = np.zeros((Nfull, N_nest, C, Nfull))

    # For each good i and nest n, identify which prices affect the shares
    for n in range(N_nest):
        nest_goods = np.where(ind2nest[:,n] == 1)[0]
        for p in nest_goods:
            c2 = ss['C_map'][p]
            for i in range(Nfull):
                for c1 in range(C):
                    dW_full[i, n, c1, p] = dW[i, n, c1, c2]
        
    # First multiply dW_full by Ω̃_inest to get dΩ_tilde
    # Expand Ω̃_inest to broadcast across country and price dimensions
    Ω̃_inest_expanded = Ω̃_inest[:,:,np.newaxis,np.newaxis]
    dΩ_tilde = dW_full * Ω̃_inest_expanded

    # Convert dΩ_tilde from (Nfull, N_nest, C, Nfull) to (Nfull, Nfull, Nfull)
    # by using nest_c_to_j mapping to aggregate over nests and countries
    dΩ_tilde_full = np.zeros((Nfull, Nfull, Nfull))
    for i in range(Nfull):
        for p in range(Nfull):
            for n in range(N_nest):
                for c in range(C):
                    j = nest_c_to_j[n, c]
                    dΩ_tilde_full[i, j, p] = dΩ_tilde[i, n, c, p]

    # Consumption goods
    # - No effect  
    # Good i 
    # Question: you shock a 
    # 
    
    return dΩ_tilde_full




def make_dOmega_tilde_dp_capital_labor(ss, capital_labor):
    C, N, K, F, Ω_tilde, N_c, K_c, F_c = ss['C'], ss['N'], ss['K'], ss['F'], ss['Ω_tilde'], ss['N_c'], ss['K_c'], ss['F_c']
    Nfull = C + N + K + F
    #N_nest = 1 + N_c + K_c + F_c
    N_nest = 1 + N_c + 1

    #θ = np.array([1] + [ss['θ'][0]]*N_c + [1]*(K_c + F_c))
    θ = np.array([1] + [ss['θ'][0]]*N_c + [capital_labor])
    

    ind2nest = np.zeros((Nfull, N_nest))
    
    ind2nest[0:3,0] = 1
    

    for n in range(N_c):
        ind2nest[slice(C + n, C + N , N_c), 1 + n] = 1

    
    ind2nest[slice(C + N, C + N + K + F), N_nest - 1] = 1
    
    # Create mapping from (nest,c) to j indices
    nest_c_to_j = np.zeros((N_nest, C), dtype=int)
    for j in range(Nfull):
        for nest in range(N_nest):
            for c in range(C):
                if ind2nest[j,nest] == 1 and ss['C_map'][j] == c:
                    nest_c_to_j[nest,c] = j

    # Construct Omega matrix mapped to nest,c dimensions
    ind2nestC = np.zeros((Nfull, N_nest, C))
    for i in range(Nfull):
        for nest in range(N_nest):
            for c in range(C):
                j = nest_c_to_j[nest,c]
                ind2nestC[i,nest,c] = Ω_tilde[i,j]

    # Sum across countries to get Ω̃_inest            
    Ω̃_inest = np.sum(ind2nestC, axis=2)

    # Create W array with normalized weights
    W = np.zeros((Nfull, N_nest, C))
    country_sums = np.sum(ind2nestC, axis=2, keepdims=True)
    nonzero_mask = (country_sums != 0)
    W = np.where(nonzero_mask, ind2nestC / country_sums, 0)

    # Create the derivative of W with respect to prices
    C = W.shape[2]
    dW = np.zeros((W.shape[0], W.shape[1], C, C))


    
    for i in range(W.shape[0]):    # Loop over goods
        for n in range(W.shape[1]): # Loop over nests
            for c1 in range(C):     # First country index
                for c2 in range(C): # Second country index
                    if c1 == c2:
                        # Diagonal term
                        dW[i,n,c1,c2] = W[i,n,c1]
                    # Subtract outer product term
                    dW[i,n,c1,c2] -= W[i,n,c1] * W[i,n,c2]
    # Multiply by (1-θ) factor                
    dW *= (1-θ[np.newaxis, :, np.newaxis, np.newaxis])

    dW_full = np.zeros((Nfull, N_nest, C, Nfull))

    # For each good i and nest n, identify which prices affect the shares
    for n in range(N_nest):
        nest_goods = np.where(ind2nest[:,n] == 1)[0]
        for p in nest_goods:
            c2 = ss['C_map'][p]
            for i in range(Nfull):
                for c1 in range(C):
                    dW_full[i, n, c1, p] = dW[i, n, c1, c2]
        
    # First multiply dW_full by Ω̃_inest to get dΩ_tilde
    # Expand Ω̃_inest to broadcast across country and price dimensions
    Ω̃_inest_expanded = Ω̃_inest[:,:,np.newaxis,np.newaxis]
    dΩ_tilde = dW_full * Ω̃_inest_expanded

    # Convert dΩ_tilde from (Nfull, N_nest, C, Nfull) to (Nfull, Nfull, Nfull)
    # by using nest_c_to_j mapping to aggregate over nests and countries
    dΩ_tilde_full = np.zeros((Nfull, Nfull, Nfull))
    for i in range(Nfull):
        for p in range(Nfull):
            for n in range(N_nest):
                for c in range(C):
                    j = nest_c_to_j[n, c]
                    dΩ_tilde_full[i, j, p] = dΩ_tilde[i, n, c, p]

    # Consumption goods
    # - No effect  
    # Good i 
    # Question: you shock a 
    # 
    
    return dΩ_tilde_full





def make_BGP_objects(region_dict, \
        closed_economy = None, \
        closed_country_index = None, \
        time_smoothing = True, \
        Ns = 27, \
        Nr = 41):
    # Load intermediate results
    path = "@Import/"
    
    with open(path + 'data_intermediate/calibration_intermediate.pkl', 'rb') as f:
        IO_95_09 = pickle.load(f)
        consumption_95_09 = pickle.load(f)
        inv_flows_njik = pickle.load(f)
        inv_flows_njik_ratio = pickle.load(f)
        GFCF_95_09 = pickle.load(f)
        VA_95_09 = pickle.load(f)
        gos_use_95_09 = pickle.load(f)
        HS_labor_C_95_09 = pickle.load(f)
        MS_labor_C_95_09 = pickle.load(f)
        LS_labor_C_95_09 = pickle.load(f)
        average_gos_GFCF_ratio = pickle.load(f)
        consumption = pickle.load(f)
        depreciation_rate = pickle.load(f)
        depreciation_rate_data = pickle.load(f)

    # region_dict has an element called "USA". Find its index.
    US_index = np.where(np.array(list(region_dict.keys())) == "USA")[0][0]
    #print(US_index)

    C = len(region_dict)
    F = 3
    N_sector = 27

    labor_95_09 = np.transpose(np.array([HS_labor_C_95_09, MS_labor_C_95_09, LS_labor_C_95_09]), (1,0,2)) ## shape: (N_C_total) x F x 5

    years = np.arange(1995, 2010)

    Ω_tilde_list = []
    consumption_new_95_09_list = []
    VA_new_95_09_list = []
    bigY_95_09_list = []
    # GFCF_new_95_09_list = []
    
    
    # Create a dict with country names as keys and values
    agg_matrix      = utils.make_aggregator_matrix(region_dict)
        
    cons_by_country = np.sum(consumption_95_09[-1,:,:], axis=0)
    agg_matrix_w    = utils.make_aggregator_matrix_weighted(region_dict, cons_by_country)
    
    '''
    IO_new_95_09: shape = C * N_c x C * N_c
    inv_flows_new_95_09: shape = C * N_c x C * N_c
    inv_flows_4D_new_95_09: shape = C x N_c x C x N_c
    consumption_new_95_09: shape = C * N_c x C
    gos_new_95_09: shape = C * N_c x 1
    VA_new_95_09: shape = C * N_c x 1
    labor_new_95_09: shape = C * N_c x F
    average_gos_GFCF_ratio_new_95_09 = C * N_c
    '''
    
    inv_flows_nk = np.array([compute_inv_flows_nk(inv_flows_njik_ratio, GFCF_95_09[i,:,:], Nr, Ns) for i in range(len(years))] )

    IO_new_95_09          = reorder_matrix_t(IO_95_09, agg_matrix)
    inv_flows_new_95_09   = reorder_matrix_t(inv_flows_nk, agg_matrix)
    VA_new_95_09          = reorder_vector_t(VA_95_09, agg_matrix)
    gos_new_95_09         = reorder_vector_t(gos_use_95_09.T, agg_matrix)
    consumption_new_95_09 = reorder_cons_t(consumption_95_09, agg_matrix)
    labor_new_95_09       = reorder_labor_t(labor_95_09, agg_matrix)
    
    average_gos_GFCF_ratio_new_95_09 = reorder_vector(average_gos_GFCF_ratio, agg_matrix_w)
    
        
    for i in range(len(years)):
    
        bigY_95_09    = construct_big_Y(C, IO_new_95_09[i], consumption_new_95_09[i], inv_flows_new_95_09[i], labor_new_95_09[:,:,i], gos_new_95_09[i], F)
        Ω_tilde_95_09 = construct_big_omega_tilde(C, IO_new_95_09[i], consumption_new_95_09[i], inv_flows_new_95_09[i], labor_new_95_09[:, :, i], gos_new_95_09[i], F) # shape: C * (1 + N + K + F) x C * (1 + N + K + F)
        Ω_tilde_list.append(Ω_tilde_95_09)
        consumption_new_95_09_list.append(consumption_new_95_09[i])
        VA_new_95_09_list.append(VA_new_95_09[i])
        bigY_95_09_list.append(bigY_95_09)

    Ω_tilde_95_09_mean = np.mean(np.array(Ω_tilde_list), axis = 0)
    bigY_95_09_list    = np.array(bigY_95_09_list)

    ########NOTE: remove education sector########
    education_index = 24
    education_indices = np.arange(C+education_index, C+C*2*N_sector, N_sector)

    Ω_tilde_95_09_mean = np.delete(Ω_tilde_95_09_mean, education_indices, axis=0) 
    Ω_tilde_95_09_mean = np.delete(Ω_tilde_95_09_mean, education_indices, axis=1) ## shape: C * (1 + 26 + 26 + F) x C * (1 + 26 + 26 + F)

    ## re-normalization
    for i in range(len(Ω_tilde_95_09_mean)):
        ### avoid division by zero
        if np.sum(Ω_tilde_95_09_mean[i, :]) == 0:
            Ω_tilde_95_09_mean[i, :] = 0
        else:
            Ω_tilde_95_09_mean[i, :] = Ω_tilde_95_09_mean[i, :] / np.sum(Ω_tilde_95_09_mean[i, :])
    μ = construct_μ_from_SEA(C, average_gos_GFCF_ratio_new_95_09, F) # shape: C * (1 + N + K + F) x 1 ## NOTE: gos and GFCF are both from SEA data
    μ_data = np.delete(μ, education_indices) ## shape: C * (1 + N + K + F) 
    N_sector = 26
    
    
    # print('μ is ', np.sum(μ_data==0))
    # μ_1 = np.ones(C * (1 + N_sector * 2 + F)) ## when δ is very big
    # ##############################
    # μ = μ_1.copy()
    μ = μ_data.copy()
    ##############################
    Ω_mean = np.divide(Ω_tilde_95_09_mean, μ[:, None], out=np.zeros_like(Ω_tilde_95_09_mean), where=μ[:, None]!=0) # shape: C * (1 + N + K + F) x C * (1 + N + K + F)

    consumption_new_95_09_list = np.array(consumption_new_95_09_list) # shape: 15 x 81 x 3
    VA_new_95_09_list = np.array(VA_new_95_09_list) # shape: 15 x 81
    # GFCF_new_95_09_list = np.array(GFCF_new_95_09_list) # shape: 15 x 3 x 27

    world_consumption_95_09 = np.sum(consumption_new_95_09_list, axis=(1, 2))
    consumption_weight_new_95_09 = np.sum(consumption_new_95_09_list, axis=1) / world_consumption_95_09[:,None] # shape: 5 x C
    ### household consumption
    Φ_c_mean = np.zeros(len(Ω_mean))
    Φ_c_mean[:C] = np.mean(consumption_weight_new_95_09, axis=0).copy()
    # print('Φ_c mean is: ', Φ_c_mean[:C])

# ###### Sanity check on the Omega matrix and see if it gives us sensible results
#     evals, evecs = np.linalg.eig(Ω_mean)
#     print('largest eval:', np.max(np.abs(evals)))
#     Ω_sum = np.eye(len(Ω_mean))
#     Ω_pow = Ω_mean.copy()
#     for i in range(1000):
#         Ω_sum += Ω_pow
#         Ω_pow  = Ω_pow @ Ω_mean
    ### Leontief inverse
    Ψ = np.linalg.inv(np.eye(len(Ω_mean))-Ω_mean) # shape: C * (1 + N + K + F) x C * (1 + N + K + F)
    Ψ_tilde = np.linalg.inv(np.eye(len(Ω_tilde_95_09_mean))-Ω_tilde_95_09_mean) # shape: C * (1 + N + K + F) x C * (1 + N + K + F)
    λ = Φ_c_mean @ Ψ # shape: C * (1 + N + K + F) x 1
    λ_tilde = Φ_c_mean @ Ψ_tilde

    # ### T_c
    N_sector = 26 ## 26
    N = C * N_sector
    F_c = 3

    ##### Production block
    ### other parameter values
    ###########################################################
    ## NOTE: change r and see how other parameter values change
    r = 0.02
    ###########################################################
    g_L = 0.0

    μ_K = μ[C+N:C+2*N]
    if np.all(μ_K!=1):
        depreciation_rate = np.delete(depreciation_rate, education_index) 
    else:
        depreciation_rate = 1000000 * np.ones(N_sector) 

    ### Calibration of g_A using US capital stock data
    consumption_US = np.sum(consumption[:,utils.get_country_index('USA')])
    # inv_US = np.sum(inv_flows_njik[:,:,39,:])
    inv_US = np.sum(inv_flows_njik[39,:,39,:]) # sum over K_c and N_c
    inv_consumption_ratio_US = inv_US / consumption_US
    #print('total investment over total consumption in the US is: ', inv_consumption_ratio_US) ## around 0.212
    # g = (inv_US - np.sum(depreciation_rate * net_stock_data_consolidated)) / (np.sum(net_stock_data_consolidated)) ## g = 0.0298
    ### alternative way of calibrating g such that K/Y = 2.7
    
    def computing_g(x, λ, μ, δ):
        eq = np.sum(λ[C+N+N_sector*US_index:C+N+(US_index+1)*N_sector] / (μ[C+N+N_sector*US_index:C+N+(US_index+1)*N_sector] \
                * (x + δ))) - 2.7 * (λ[US_index] + np.sum(λ[C+N+N_sector*US_index:C+N+(US_index+1)*N_sector] / μ[C+N+N_sector*US_index:C+N+(US_index+1)*N_sector]))
        return eq

    initial_guess = 0.02


    if np.any(μ_K!=1):
        # Solve the system of equations numerically
        sol = opt.root(computing_g, initial_guess, args=(λ, μ, depreciation_rate))
        g = sol.x ## 
    else:
        g = 0.0166 ##0.0188
    g_A = g - g_L
    #print('g is: ', g)

    #### Calibrate r_i for every sector and country
    δ_i = np.tile(depreciation_rate, C) 


    δ_i_data = np.tile(np.delete(depreciation_rate_data, education_index), C) 
    r_i = μ[C+N:C+2*N] * (δ_i_data + g) - δ_i_data ## ISSUE: r_i is negative for US education sector

    ### Back out B_c and b_c (net foreign asset positions relative to world consumption) 
    # assuming T_c_tilde = 0
    # b_c = T_c / (r-g)
    # B_c = np.mean(world_consumption_95_09) * b_c
    

    #### Alternative way of calibrate b_c (using EWN data) [positive B_c means net debtor in the international market]
    NFA_data = pd.read_excel(path + 'data_raw/EWN-dataset.xlsx',sheet_name='Dataset')
    
    country_list = np.asarray(NFA_data.iloc[:,0].unique()) # 214 countries
    years_list   = np.asarray(NFA_data.iloc[:,2].unique())  # 53 years [1970-2022]
    net_IIP      = np.asarray(NFA_data.iloc[:,14]) ### Net IIP excl gold
    
    
    ### extract relevant information
    net_IIP_2D = rearrange(net_IIP, '(c t) -> c t', c = len(country_list))
    ### only keep years 1995-2009
    IIP_country_indices = np.array([10, 11, 18, 29, 26, 34, 40, 50, 51, 74, 52, 177, 62, 68, 69, 203, 77, 87, 90, 89, 93, 96, 98, 104, 115, 116, 109, 126, 122, 138, 155, 156, 158, 159, 171, 172, 184, 196, 187, 204])

    net_IIP_2D = net_IIP_2D[IIP_country_indices , 25:40]
    # Replace nan with 0
    net_IIP_2D[np.isnan(net_IIP_2D)] = 0
    # For each year add an ROW country
    net_IIP_2D = np.concatenate([net_IIP_2D, np.zeros((1, net_IIP_2D.shape[1]))], axis=0)
    # For each country, set that to negative of the sum of all other countries
    net_IIP_2D[-1,:] = -np.sum(net_IIP_2D[:-1,:], axis=0)

    B_c_new = agg_matrix @ net_IIP_2D
    
    GFCF_new = np.einsum('ix,tnx -> ti', agg_matrix, GFCF_95_09)
    b_c_new = np.mean(B_c_new / (np.sum(VA_new_95_09_list, axis=1) \
                                 - np.sum(GFCF_new, axis=1))[None,:], axis=1)
    T_c_new = b_c_new * (r-g)
    
    # Φ_c_derived = np.mean((np.sum(np.reshape(VA_new_95_09_list, (len(years), C, N_sector+1)), axis=2) - GFCF_new + (r-g) * B_c_new.T) / (np.sum(VA_new_95_09_list, axis=1) - np.sum(GFCF_new, axis=1))[:,None], axis=0)
    ### Compute a fixed point problem to compute Φ_c_derived
    temp_mat = np.zeros((C+N+N+F*C, C))
    for c in range(C):
        temp_mat[:,c] = np.sum(Ψ[:,C+N+N+F_c*c:C+N+N+F_c*(c+1)], axis=1)
        
        # Get the relevant slice of Ψ
        psi_slice = Ψ[:,C+N+N_sector*c:C+N+N_sector*(c+1)]
        
        # Get the relevant μ terms and reshape to match psi_slice columns
        mu_terms = (1 - 1/μ[C+N+N_sector*c:C+N+N_sector*(c+1)])
        mu_terms = mu_terms.reshape(1, -1)  # Make row vector to match columns
        
        # Multiply each column of Ψ by corresponding μ term and sum
        temp_mat[:,c] = np.sum(psi_slice * mu_terms, axis=1) + temp_mat[:,c]
        
    M = temp_mat[:C, :C].copy()

    # Take (I - M)'
    M = (np.eye(C) - M).T

    # Add new row with all ones
    M = np.vstack([M, np.ones([1, C])])

    # Create modified T vector
    T_modified = T_c_new.copy()
    # Add constraint that Φ sums to 1
    T_modified = np.append(T_modified, 1)
    Φ_c_derived = np.linalg.inv(M.T @ M) @ M.T @ T_modified
    
    Φ_c_derived_static = np.mean((np.sum(np.reshape(VA_new_95_09_list, (len(years), C, N_sector+1)), axis=2) + (r-g) * B_c_new.T) / (np.sum(VA_new_95_09_list, axis=1))[:,None], axis=0)

    Φ_c_derived = np.concatenate([Φ_c_derived, np.zeros(2*N+C*F)])  ### compare with Φ_c_mean
    λ_derived = Φ_c_derived @ Ψ
    λ_tilde_derived = Φ_c_derived @ Ψ_tilde

    ### Pick the calibration way to compute b_c, T_c, λ, Φ_c
    b_c = b_c_new.copy()
    B_c = B_c_new.copy()
    T_c = T_c_new.copy()
    λ = λ_derived.copy()
    λ_tilde = λ_tilde_derived.copy()
    Φ_c_mean = Φ_c_derived.copy()
    Φ = Φ_c_mean.copy()

    ####################################
    #  Pick the option of λ, λ_tilde, Φ_c to use

    # λ = λ.copy()
    # λ_tilde = λ_tilde.copy()
    # Φ_c_mean = Φ_c_mean.copy()
    ####################################
       
    # Φ_c_mean = Φ_c_mean.copy()
    
    ### Financial markets
    ψ_US = 0.65 
    γ = 0.5
    ϵ_d = 40 * γ - 2

    def compute_ρ_c(g_ωc, S_c, γ, r):
        return r + ((γ + 1) / 2) * (S_c ** 2) - g_ωc / γ

    def sub_system_equations(x, c):
        '''
        solving the system of equations country by country
        x[0]: ν_c (shape: C x 1)
        x[1]: S_c (shape: C x 1)
        x[2]: g_ωc (shape: C x 1)
        '''

        χ_c = (x[0] + g_L) / (x[0] + g - x[2]) # scalar

        ### equation 1 on ν_c
        eq1 = ϵ_d + 1/(r + x[0] - g_A) - χ_c * γ / (x[2] - g_A) 

        ### equation 2 on S_c
        eq2 = np.sum(np.reshape((r_i - r) * λ[C+N:C+2*N] / (r_i + δ_i), (C, N_sector)), axis=1)[c] / np.sum(np.reshape(λ[C+2*N:], (C, F_c)), axis=1)[c] - χ_c * γ * (x[1] ** 2) / (r + x[0] - g_A)

        ### equation 3 on g_ωc
        eq3 = b_c[c] + np.sum(np.reshape(λ[C+N:C+2*N] / (r_i + δ_i), (C, N_sector)), axis=1)[c] / np.sum(np.reshape(λ[C+2*N:], (C, F_c)), axis=1)[c] - (χ_c - 1) / (r + x[0] - g_A)

        list_eqs = np.concatenate([eq1.flatten(), eq2.flatten(), eq3.flatten()])
        return list_eqs

    # Provide initial guesses for the unknowns
    ν_c_list = []
    S_c_list = []
    g_ωc_list = []
    ρ_c_list = []
    χ_c_list = []
    sanity_checks = []
    
    for c in range(C):
        initial_guess = np.array([0.05, 0.05, 0.05]) # 0.05 * np.ones(3)
        # initial_guess = np.array([[0.01, 0.01, 0.1], [-0.1, -0.1, -0.1], [-0.1, -0.1, -0.1]])
        # Solve the system of equations numerically
        sol = opt.root(sub_system_equations, initial_guess, args=(c))
        sanity_checks.append(sol.success)
        ν_c, S_c, g_ωc = sol.x 
        S_c = np.abs(S_c)
        ρ_c = compute_ρ_c(g_ωc, S_c, γ, r)
        χ_c = (ν_c + g_L) / (ν_c + g - g_ωc)
        ν_c_list.append(ν_c)
        S_c_list.append(S_c)
        g_ωc_list.append(g_ωc)
        ρ_c_list.append(ρ_c)
        χ_c_list.append(χ_c)

    ### Sanity checks
    all_checks = [
        np.all(np.array(ρ_c_list) + np.array(ν_c_list) > 0),  # Check ρ_c + ν_c > 0
        np.all(sanity_checks)  # Check all solutions converged
    ]
    print('Passes sanity checks:', np.all(all_checks))

    ### calibrate σ_i and ψ_i for other countries
    r_US = r_i[US_index*N_sector:(US_index+1)*N_sector] ## shape: N_sector x 1
    S_c_US = S_c_list[US_index] ## shape: scalar

    σ_US = (r_US - r) / (S_c_US * ψ_US) ## shape: N_sector x 1
    σ_i = np.tile(σ_US, C) ## shape: C * N_sector x 1

    S_c_repeat = np.repeat(np.array(S_c_list), N_sector) ## shape: C * N_sector x 1
    ψ_i = (r_i - r) / (S_c_repeat * σ_i) ## shape: C * N_sector x 1

    ### Report and analysis
    # ψ_i_mean = np.mean(np.reshape(ψ_i, (C, N_sector)), axis=1)
    # ψ_i_sd = np.std(np.reshape(ψ_i, (C, N_sector)), axis=1)
    # σ_US
    # r_i_mean = np.mean(np.reshape(r_i, (C, N_sector)), axis=1)
    # r_i_sd = np.std(np.reshape(r_i, (C, N_sector)), axis=1)

    #### closed-economy case
    if closed_economy:
        new_C = 1
        original_N_sector = 27
        new_Y_95_09_list = np.zeros((new_C+original_N_sector*2+F, new_C+original_N_sector*2+F, 15))
        new_Ω_tilde_95_09 = np.zeros((new_C+original_N_sector*2+F, new_C+original_N_sector*2+F, 15))
        for i in range(len(years)):
            new_Y_95_09_list[:,:,i] = \
                closed_economy_collapse(closed_country_index, bigY_95_09_list[i,:,:], C, original_N_sector, original_N_sector, F)
            new_Ω_tilde_95_09[:,:,i] = \
                np.divide(new_Y_95_09_list[:,:,i], np.sum(new_Y_95_09_list[:,:,i], axis=1)[:,None], \
                          out=np.zeros_like(new_Y_95_09_list[:,:,i]), where=np.sum(new_Y_95_09_list[:,:,i], axis=1)[:,None]!=0)
        #### replace variable names
        Ω_tilde_95_09_mean = np.mean(np.array(new_Ω_tilde_95_09), axis = 2)
        old_C = C
        C = 1
        education_indices = np.arange(C+education_index, C+C*2*N_sector, N_sector)
        Ω_tilde_95_09_mean = np.delete(Ω_tilde_95_09_mean, education_indices, axis=0) 
        Ω_tilde_95_09_mean = np.delete(Ω_tilde_95_09_mean, education_indices, axis=1)
        ### replace negative consumption to zero
        Ω_tilde_95_09_mean[0, :][Ω_tilde_95_09_mean[0, :] < 0] = 0
        ### re-normalization
        for i in range(len(Ω_tilde_95_09_mean)):
        ### avoid division by zero
            if np.sum(Ω_tilde_95_09_mean[i, :]) == 0:
                Ω_tilde_95_09_mean[i, :] = 0
            else:
                Ω_tilde_95_09_mean[i, :] = Ω_tilde_95_09_mean[i, :] / np.sum(Ω_tilde_95_09_mean[i, :])
        ### replace other variables
        N = N_sector * C
        K = N_sector * C
        F = F_c * C
        μ = np.concatenate(([μ[closed_country_index]], \
                             μ[old_C+N_sector*closed_country_index: old_C+N_sector*(closed_country_index+1)], \
                            μ[old_C+old_C*N_sector+N_sector*closed_country_index: old_C+old_C*N_sector+N_sector*(closed_country_index+1)], \
                            μ[old_C+2*old_C*N_sector+F*closed_country_index: old_C+2*old_C*N_sector+F*(closed_country_index+1)]))
        Ω_mean = np.divide(Ω_tilde_95_09_mean, μ[:, None], \
                           out=np.zeros_like(Ω_tilde_95_09_mean), \
                        where=μ[:, None]!=0)
        Ψ = np.linalg.inv(np.eye(len(Ω_mean))-Ω_mean)
        Ψ_tilde = np.linalg.inv(np.eye(len(Ω_tilde_95_09_mean))-Ω_tilde_95_09_mean)
        Φ = np.zeros(C + N + K + F)
        Φ_c_mean = np.ones(C)
        Φ[0] = Φ_c_mean
        λ = Φ @ Ψ
        λ_tilde = Φ @ Ψ_tilde
        r_i = r_i[N_sector*closed_country_index: N_sector*(closed_country_index+1)]
        ψ_i = ψ_i[N_sector*closed_country_index: N_sector*(closed_country_index+1)]
        σ_i = σ_i[N_sector*closed_country_index: N_sector*(closed_country_index+1)]
        δ_i = δ_i[N_sector*closed_country_index: N_sector*(closed_country_index+1)]
        S_c_list = [S_c_list[closed_country_index]]
        χ_c_list = [χ_c_list[closed_country_index]]
        g_ωc_list = [g_ωc_list[closed_country_index]]
        ρ_c_list = [ρ_c_list[closed_country_index]]
        ν_c_list = [ν_c_list[closed_country_index]]
        b_c = np.zeros(C)
        B_c = np.zeros(C)
        T_c = np.zeros(C)
        bigY_95_09_list = new_Y_95_09_list.copy()

    #print('There are countries: ',C)

    #########################
    # Export parameter values
    #########################
    
    
    
    parameters = {'Omega': Ω_mean, 'Omega_tilde': Ω_tilde_95_09_mean, 'Psi': Ψ, 'Psi_tilde': Ψ_tilde, 'mu': μ, \
            'delta': δ_i, 'g_A': g_A, 'g_L': g_L, 'g': g, 'r_i': r_i, 'r': r, 'rho_c': ρ_c_list, 'S_c': S_c_list, \
            'psi': ψ_i, 'sigma': σ_i, 'chi_c': χ_c_list, 'g_omega': g_ωc_list, 'nu_c': ν_c_list, 'Phi': Φ, \
            'Phi_static': Φ_c_derived_static, 'gamma': γ, 'epsilon_d': ϵ_d, 'B_c': B_c, 'b_c': b_c, 'T_c': T_c, \
            'lambda': λ, 'lambda_tilde': λ_tilde, 'C': C, 'selected_countries': region_dict}


    if time_smoothing == False:
        raw_data = {'education_index':education_indices, 'C': C, 'N': N, 'F': F}
    else:
        raw_data = {'bigY': bigY_95_09_list, 'education_index':education_indices, 'C': C, 'N': N, 'F': F}
        
    return parameters, raw_data

    




    
def compute_inv_flows_nk(inv_flows_njik_ratio, GFCF, Nr, Ns):
        GFCF_nk = np.sum(np.reshape(GFCF, (Nr, Ns, Nr)), axis=0).T
        new_inv_flows_njik = GFCF_nk[:,None,None,:] * inv_flows_njik_ratio
        # new_inv_flows_nk = np.reshape(new_inv_flows_njik, (Nr*Ns, Nr*Ns))
        # print('method 1 is: ', np.sum(np.reshape(new_inv_flows_njik, (Nr*Ns, Nr*Ns)), axis=0))
        new_inv_flows_nk = np.reshape(np.transpose(new_inv_flows_njik, (2, 1, 0, 3)), (Nr*Ns, Nr*Ns))
        # print('method 2 is: ', np.sum(np.reshape(np.transpose(new_inv_flows_njik, (2, 1, 0, 3)), (Nr*Ns, Nr*Ns)), axis=0))
        return new_inv_flows_nk


def reorder_matrix_t(matrix, agg_matrix):
    C, Nr = agg_matrix.shape
    matrix = rearrange(matrix, "t (i m) (j n) -> t i j m n", i = Nr, j = Nr)
    matrix = np.einsum('ix,jy,txymn->tijmn', agg_matrix, agg_matrix, matrix)
    matrix = rearrange(matrix, "t i j m n -> t (i m) (j n)", i = C, j = C)
    return matrix

def reorder_cons_t(matrix, agg_matrix):
    C, Nr = agg_matrix.shape
    matrix   = rearrange(matrix, "t (i n) j -> t i j n", i = Nr, j = Nr)
    matrix   = np.einsum('ix,jy,txyn->tijn', agg_matrix, agg_matrix, matrix)
    matrix   = rearrange(matrix, "t i j n -> t (i n) j", i = C, j = C)
    return matrix 

def reorder_vector_t(vector, agg_matrix):
    C, Nr = agg_matrix.shape
    vector = rearrange(vector, "t (i m) ->t i m", i = Nr)
    vector = np.einsum('ix,txm->tim', agg_matrix, vector)
    vector = rearrange(vector, "t i m -> t (i m)", i = C)
    return vector

def reorder_vector(vector, agg_matrix):
    C, Nr = agg_matrix.shape
    vector = rearrange(vector, "(i m) -> i m", i = Nr)
    vector = np.einsum('ix,xm->im', agg_matrix, vector)
    vector = rearrange(vector, "i m -> (i m)", i = C)
    return vector


def reorder_labor_t(labor, agg_matrix):
    C, Nr = agg_matrix.shape
    labor = rearrange(labor, "(i n) f t -> i n f t", i = Nr)
    labor = np.einsum('ix, xnft-> inft', agg_matrix, labor)
    labor = rearrange(labor, "i n f t -> (i n) f t", i = C)
    return labor


def construct_big_omega_tilde(C, IO, consumption, inv_flows, labor, gos, F):
    N = len(IO)
    Omega_tilde = np.zeros((C + 2 * N + C*F, C + 2 * N + C*F))
    big_Y = construct_big_Y(C, IO, consumption, inv_flows, labor, gos, F)
    for i in range(len(big_Y)):
        ## avoid division by zero problem
        if np.sum(big_Y[i, :]) == 0:
            Omega_tilde[i, :] = 0
        else:
            Omega_tilde[i, :] = big_Y[i, :] / np.sum(big_Y[i, :])

    return Omega_tilde


def construct_big_Y(C, IO, consumption, inv_flows, labor, gos, F):
    N = len(IO)
    Y = np.zeros((C + 2 * N + C*F, C + 2 * N + C*F))
    #################
    # consumption rows
    #################
    Y[:C,C:C+N] = consumption.T 
    #################
    # production rows
    #################
    Y[C:C+N, C:C+N] = IO # Ynn
    Y[C:C+N, C+N:C+2*N] = np.diag(gos) # Ynk
    #################
    # primary factor columns (block diagonal matrix)
    #################
    # Create a N x C*F block diagonal matrix
    block_diagonal_matrix = np.zeros((N, C*F))

    # Assign values to the block diagonal matrix
    N_sector = 27
    for i in range(C):
        start_row = i * N_sector
        end_row = (i + 1) * N_sector
        block_diagonal_matrix[start_row:end_row, i*F:(i+1)*F] = labor[start_row:end_row, :]

    Y[C:C+N, C+2*N:] = block_diagonal_matrix # Ynf
    #################
    # capital aggregator rows
    #################
    Y[C+N:C+2*N, C:C+N] = inv_flows # Ykn (reason: to fit the depreciation data better)
    # print('last few row sum for goods are', np.sum(Y,axis=1)[3+54:3+81])
    
    return Y


def construct_μ_from_SEA(C, average_gos_GFCF_ratio, F):
    N = 27
    μ = np.ones(C + 2 * C * N + C * F)
    
    μ[C+C*N:C+2*C*N] = average_gos_GFCF_ratio.copy()

    return μ

def closed_economy_collapse(country_index, bigY, C, N_c, K_c, F_c):
    new_C = 1
    N = C * N_c
    K = C * K_c
    F = C * F_c
    collapsed_Y = np.zeros((new_C+N_c+K_c+F_c, new_C+N_c+K_c+F_c))
    #### consumption part
    # collapsed_Y[0,new_C:new_C+N_c] = \
    #     np.reshape(bigY[country_index, C:C+N], (C, N_c)).sum(0)
    #### regular goods part
    collapsed_Y[new_C:new_C+N_c,new_C:new_C+N_c] = \
        np.reshape(bigY[C+country_index*N_c:C+(country_index+1)*N_c, C:C+N], (N_c, C, N_c)).sum(1)
    ### investment goods part
    collapsed_Y[new_C+N_c:new_C+N_c+K_c,new_C:new_C+N_c] = \
        np.reshape(bigY[C+N+country_index*K_c:C+N+(country_index+1)*K_c, C:C+N], (K_c, C, N_c)).sum(1)
    ### primary factors part
    collapsed_Y[new_C:new_C+N_c,-F_c:] = \
        np.reshape(bigY[C+country_index*N_c:C+(country_index+1)*N_c, -F:], (N_c, C, F_c)).sum(1)
    ### investment part
    collapsed_Y[new_C:new_C+N_c,-F_c-K_c:-F_c] = \
        np.reshape(bigY[C+country_index*N_c:C+(country_index+1)*N_c, -F-K:-F], (N_c, C, K_c)).sum(1)
    ### net export and final consumption part
    for i in range(N_c):
        collapsed_Y[0,new_C+i] = np.sum(collapsed_Y[new_C+i,:]) - np.sum(collapsed_Y[1:,new_C+i])
    return collapsed_Y




def make_dOmega_tilde_dp(ss):
    C, N, K, F, Ω_tilde, N_c, K_c, F_c = ss['C'], ss['N'], ss['K'], ss['F'], ss['Ω_tilde'], ss['N_c'], ss['K_c'], ss['F_c']
    Nfull = C + N + K + F
    N_nest = 1 + N_c + K_c + F_c

    θ = np.array([1] + [ss['θ'][0]]*N_c + [1]*(K_c + F_c))

    ind2nest = np.zeros((Nfull, N_nest))
    ind2nest[0:3,0] = 1

    for n in range(N_c):
        ind2nest[slice(C + n, C + N , N_c), 1 + n] = 1

    for k in range(K_c):
        ind2nest[slice(C + N + k, C + N + K, K_c), 1 + N_c + k] = 1

    for f in range(F_c):
        ind2nest[slice(C + N + K + f, C + N + K + F, F_c), 1 + N_c + K_c + f] = 1

    # Create mapping from (nest,c) to j indices
    nest_c_to_j = np.zeros((N_nest, C), dtype=int)
    for j in range(Nfull):
        for nest in range(N_nest):
            for c in range(C):
                if ind2nest[j,nest] == 1 and ss['C_map'][j] == c:
                    nest_c_to_j[nest,c] = j

    # Construct Omega matrix mapped to nest,c dimensions
    ind2nestC = np.zeros((Nfull, N_nest, C))
    for i in range(Nfull):
        for nest in range(N_nest):
            for c in range(C):
                j = nest_c_to_j[nest,c]
                ind2nestC[i,nest,c] = Ω_tilde[i,j]

    # Sum across countries to get Ω̃_inest            
    Ω̃_inest = np.sum(ind2nestC, axis=2)

    # Create W array with normalized weights
    W = np.zeros((Nfull, N_nest, C))
    country_sums = np.sum(ind2nestC, axis=2, keepdims=True)
    nonzero_mask = (country_sums != 0)
    W = np.divide(ind2nestC, country_sums, out=np.zeros_like(ind2nestC), where=nonzero_mask)

    # Create the derivative of W with respect to prices
    C = W.shape[2]
    dW = np.zeros((W.shape[0], W.shape[1], C, C))

    for i in range(W.shape[0]):    # Loop over goods
        for n in range(W.shape[1]): # Loop over nests
            for c1 in range(C):     # First country index
                for c2 in range(C): # Second country index
                    if c1 == c2:
                        # Diagonal term
                        dW[i,n,c1,c2] = W[i,n,c1]
                    # Subtract outer product term
                    dW[i,n,c1,c2] -= W[i,n,c1] * W[i,n,c2]
    # Multiply by (1-θ) factor                
    dW *= (1-θ[np.newaxis, :, np.newaxis, np.newaxis])

    dW_full = np.zeros((Nfull, N_nest, C, Nfull))

    # For each good i and nest n, identify which prices affect the shares
    for n in range(N_nest):
        nest_goods = np.where(ind2nest[:,n] == 1)[0]
        for p in nest_goods:
            c2 = ss['C_map'][p]
            for i in range(Nfull):
                for c1 in range(C):
                    dW_full[i, n, c1, p] = dW[i, n, c1, c2]
        
    # First multiply dW_full by Ω̃_inest to get dΩ_tilde
    # Expand Ω̃_inest to broadcast across country and price dimensions
    Ω̃_inest_expanded = Ω̃_inest[:,:,np.newaxis,np.newaxis]
    dΩ_tilde = dW_full * Ω̃_inest_expanded

    # Convert dΩ_tilde from (Nfull, N_nest, C, Nfull) to (Nfull, Nfull, Nfull)
    # by using nest_c_to_j mapping to aggregate over nests and countries
    dΩ_tilde_full = np.zeros((Nfull, Nfull, Nfull))
    for i in range(Nfull):
        for p in range(Nfull):
            for n in range(N_nest):
                for c in range(C):
                    j = nest_c_to_j[n, c]
                    dΩ_tilde_full[i, j, p] = dΩ_tilde[i, n, c, p]

    return dΩ_tilde_full




@njit
def make_dOmega_tilde_dp_fast(C, N, K, F, Ω_tilde, N_c, K_c, F_c, θ, C_map):
    Nfull = C + N + K + F
    N_nest = 1 + N_c + K_c + F_c
    
    # Pre-allocate arrays
    ind2nest = np.zeros((Nfull, N_nest))
    ind2nest[0:3,0] = 1
    
    # Vectorized slice assignments
    for n in range(N_c):
        ind2nest[C + n:C + N:N_c, 1 + n] = 1
    for k in range(K_c):
        ind2nest[C + N + k:C + N + K:K_c, 1 + N_c + k] = 1
    for f in range(F_c):
        ind2nest[C + N + K + f:C + N + K + F:F_c, 1 + N_c + K_c + f] = 1

    # Pre-compute nest_c_to_j mapping
    nest_c_to_j = np.zeros((N_nest, C), dtype=np.int32)
    for j in range(Nfull):
        for nest in range(N_nest):
            if ind2nest[j,nest]:
                nest_c_to_j[nest,C_map[j]] = j

    # Construct ind2nestC directly using nest_c_to_j
    ind2nestC = np.zeros((Nfull, N_nest, C))
    for i in range(Nfull):
        for nest in range(N_nest):
            for c in range(C):
                ind2nestC[i,nest,c] = Ω_tilde[i,nest_c_to_j[nest,c]]

    # Compute W array
    Ω̃_inest = np.sum(ind2nestC, axis=2)
    country_sums = np.sum(ind2nestC, axis=2)
    country_sums = country_sums.reshape(Nfull, N_nest, 1)
    W = np.zeros((Nfull, N_nest, C))
    for i in range(Nfull):
        for n in range(N_nest):
            if country_sums[i,n,0] != 0:
                W[i,n,:] = ind2nestC[i,n,:] / country_sums[i,n,0]

    # Compute dW directly 
    dW = np.zeros((Nfull, N_nest, C, C))
    for i in range(Nfull):
        for n in range(N_nest):
            w = W[i,n]
            for c1 in range(C):
                for c2 in range(C):
                    dW[i,n,c1,c2] = -w[c1] * w[c2]
                    if c1 == c2:
                        dW[i,n,c1,c2] += w[c1]
    
    dW *= (1-θ[np.newaxis, :, np.newaxis, np.newaxis])

    # Compute final result directly
    dΩ_tilde_full = np.zeros((Nfull, Nfull, Nfull))
    for n in range(N_nest):
        nest_goods = np.where(ind2nest[:,n] > 0)[0]
        for p in nest_goods:
            c2 = C_map[p]
            for i in range(Nfull):
                for c1 in range(C):
                    j = nest_c_to_j[n,c1]
                    dΩ_tilde_full[i,j,p] = dW[i,n,c1,c2] * Ω̃_inest[i,n]

    return dΩ_tilde_full

def test_dOmega_tilde_dp(ss):
    # Run both versions
    result1 = make_dOmega_tilde_dp(ss)
    
    # Extract needed parameters for fast version
    C, N, K, F = ss['C'], ss['N'], ss['K'], ss['F']
    Ω_tilde, N_c, K_c, F_c = ss['Ω_tilde'], ss['N_c'], ss['K_c'], ss['F_c']
    θ = np.array([1] + [ss['θ'][0]]*N_c + [1]*(K_c + F_c))
    C_map = np.array(ss['C_map'])
    
    result2 = make_dOmega_tilde_dp_fast(C, N, K, F, Ω_tilde, N_c, K_c, F_c, θ, C_map)
    
    
    # Compare results
    if not np.allclose(result1, result2, rtol=1e-10, atol=1e-10):
        print("Warning: Results differ!")
        print("Max difference:", np.max(np.abs(result1 - result2)))
        return False
    return True



import numpy as np



def get_calibration_objects(ss, iso_list):
    
    outcomes = {}
    
    C, N, K, F, K_c = ss['C'], ss['N'], ss['K'], ss['F'], ss['K_c']

    cap_idx   = range(C + N, C + N + K)

    
    for iso in iso_list:
        idx = ss['countries'].index(iso)

        cap_idx_c = [i for i in cap_idx if ss['C_map'][i] == idx]

        r_i_c         = np.array([ss['r_i'][k] for  k in range(K) if ss['C_map'][k + C + N] == idx])
        μ_i_c         = np.array([ss['μ'][k] for  k in range(C + N, C + N + K ) if ss['C_map'][k] == idx])

        δ_c           = np.array([ss['δ'][k] for k in range(K) if ss['C_map'][k + C + N] == idx])
        stock_weights = ss['λ'][cap_idx_c] /(r_i_c  + δ_c) / np.sum(ss['λ'][cap_idx_c] /(r_i_c  + δ_c))

        GDP = ss['Φ_c'][idx] +   np.sum( ss['λ'][cap_idx_c]/ss['μ'][cap_idx_c]) 

        inv_weights = (ss['λ'][cap_idx_c]  / ss['μ'][cap_idx_c]) / np.sum(ss['λ'][cap_idx_c] / ss['μ'][cap_idx_c])

        r_i_bar     = np.sum(stock_weights * r_i_c)

        mu_bar      = np.sum(inv_weights * μ_i_c)
        inv_rate    =  np.sum( ss['λ'][cap_idx_c]/ss['μ'][cap_idx_c])  / GDP

        cap_income  = np.sum( ss['λ'][cap_idx_c])  / GDP



        imports     = trade_flows(ss, orig = None, dest = [iso]) / GDP

        tb          = trade_flows(ss, orig = [iso], dest = None) / GDP - imports 

        nfa         = ss['b_c'][idx] / GDP
        
        outcomes[iso] = {'r_i_bar': r_i_bar, 'inv_rate': inv_rate, 'cap_income': cap_income, 'imports': imports, 'tb': tb, 'nfa' : nfa, 'mu_bar' : mu_bar}
        
    return outcomes



def get_calibration_objects2(ss, iso_list):
    
    outcomes = {}
    
    
    
    for iso in iso_list:
        idx = ss['countries'].index(iso)
        C, N, K, F, K_c = ss['C'], ss['N'], ss['K'], ss['F'], ss['K_c']



        cap_idx   = [i for i in range(C + N, C + N + K) if ss['C_map'][i] == idx]
        cap_idx_c = [i for i in cap_idx if ss['C_map'][i] == idx]

        r_i_c         = np.array([ss['r_i'][k] for  k in range(K) if ss['C_map'][k + C + N] == idx])
        μ_i_c         = np.array([ss['μ'][k] for  k in range(C + N, C + N + K ) if ss['C_map'][k] == idx])

        δ_c           = np.array([ss['δ'][k] for k in range(K) if ss['C_map'][k + C + N] == idx])
        stock_weights = ss['λ'][cap_idx_c] /(r_i_c  + δ_c) / np.sum(ss['λ'][cap_idx_c] /(r_i_c  + δ_c))

        GDP = ss['Φ_c'][idx] +   np.sum( ss['λ'][cap_idx_c]/ss['μ'][cap_idx_c]) 

        inv_weights = (ss['λ'][cap_idx_c]  / ss['μ'][cap_idx_c]) / np.sum(ss['λ'][cap_idx_c] / ss['μ'][cap_idx_c])

        r_i_bar     = np.sum(stock_weights * r_i_c)

        mu_bar      = np.sum(inv_weights * μ_i_c)
        inv_rate    =  np.sum( ss['λ'][cap_idx_c]/ss['μ'][cap_idx_c])  / GDP

        cap_income  = np.sum( ss['λ'][cap_idx_c])  / GDP



        imports     = trade_flows(ss, orig = None, dest = [iso]) / GDP

        tb          = trade_flows(ss, orig = [iso], dest = None) / GDP - imports 

        nfa         = ss['b_c'][idx] / GDP
        
        outcomes[iso] = {'r_i_bar': r_i_bar, 'inv_rate': inv_rate, 'cap_income': cap_income, 'imports': imports, 'tb': tb, 'nfa' : nfa, 'mu_bar' : mu_bar}
        
    return outcomes

    


def trade_flows(ss, orig = None, dest = None, bilateral  = False):  # This is for an import tariff
    

    # Country key  Key = C x (C + N + K + F) matrix saying with Key_{ij}=1 if  j pertains to i
    C, N, K, F = ss['C'], ss['N'], ss['K'], ss['F']
    # Get country indices from country names
    cs = list(ss['countries'])
    
    orig_idx = np.array([cs.index(iso)  for iso in orig]) if orig is not None else np.arange(C)
    dest_idx = np.array([cs.index(iso) for iso in dest]) if dest is not None else np.arange(C)

    if orig is not None and not all(c in cs for c in orig):
        raise ValueError(f"All export countries must be in: {cs}")
    if dest is not None and not all(c in cs for c in dest):
        raise ValueError(f"All import countries must be in: {cs}")
    
    trade = 0

    for i in np.arange(C + N + K):
        for j in np.arange(C + N + K):
            if ss['C_map'][i] in dest_idx and ss['C_map'][j] in orig_idx and ss['C_map'][i] != ss['C_map'][j]:
                trade += ss['λ'][i] * ss['Ω'][i,j]
    
    
    if bilateral:
        trade += trade_flows(ss, orig = dest, dest = orig, bilateral = False)
        return trade
    
    
    return trade
