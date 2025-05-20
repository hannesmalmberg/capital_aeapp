'''
Define the system of log-lienarized equations and supporting functions in the Blanchard model with taxes (multi-country case)
Author: Yutong Zhong
'''

import numpy as np
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

from scipy.sparse.linalg import LinearOperator, gcrotmk

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)



def solve(ss, shocks, full = True):
    print("Solving counterfactual...")
    
    shocks = populate_shocks(ss, shocks)
    C, F, = [ss[k] for k in ['C', 'F']]
    # Solve the system of equations in a matrix form
    n_params = sum([C, C, C, C, C, F, C, 1, C, C, C])

    Axmb = log_linearized_fun
    b = -Axmb(np.zeros(n_params), shocks, ss, full)
    def get_Ax(shocks, ss, full):
            def _Ax(x0):
                return Axmb(x0, shocks, ss, full) + b
            return _Ax

    Ax = get_Ax(shocks, ss, full)
    Ax_LO = LinearOperator([n_params, n_params], matvec=Ax)
    sol = gcrotmk(Ax_LO, b, atol=1e-10)[0]
    
    
    g, λ, Φ, g_ωc, χ_c = [ss[k] for k in ['g', 'λ', 'Φ', 'g_ωc', 'χ_c']]
    
    
    sum_dλ_v1, sum_dλ_v2, sum_dλ_v3, sum_dλ_v4, sum_dλ_v5, \
        dΛ_f, dS_c, dr, db_c, dD_c, dS_c_tilde = \
            unpack_x(sol, ss)
    
    dlogA = shocks['dlogA']
    
    
    
 
    dr_i = compute_dr_i(dr, dS_c, shocks, ss)
    dlogμ = compute_dlogμ(dr_i, shocks, ss) 
    dlogp = compute_dlogp(dΛ_f, dlogA, dlogμ, shocks, ss)
    dΦ = compute_dΦ(dr_i, dΛ_f, dD_c, db_c, sum_dλ_v1, dr, shocks, ss)
    dlogΦ = np.divide(dΦ, Φ, out=np.zeros_like(dΦ), where=Φ!=0)
    dlogg_ωc = compute_dlogg_ωc(dr, dS_c, shocks, ss)
    dlogχ_c = compute_dlogχ_c(dlogg_ωc, shocks, ss)
    dg_ωc = g_ωc * dlogg_ωc
    dχ_c = χ_c * dlogχ_c

    
    # Solve the system of equations numerically
    # initial_guess = np.zeros(9*C + F + 1)
    # sol = opt.root(Blanchard_functions.log_linearized_fun, \
    #                initial_guess, args=(shocks, ss, full),tol = 10**(-10))
    # print('The system of equations converge: ', sol.success)
    # sol = sol.x
    # sum_dλ_v1, sum_dλ_v2, sum_dλ_v3, sum_dλ_v4, sum_dλ_v5, dΛ_f, dS_c, dr, db_c, dD_c, dS_c_tilde \
    #     = Blanchard_functions.unpack_x(sol, ss)
    
   
    F_idx = slice(ss['C']+ss['N']+ss['K'], ss['C']+ss['N']+ss['K'] + ss["F"])
    dual = ss.get("dual", 0)


    # if 'dΩ_tilde_dp' in ss:
        
    #     dΩ_tilde  = (np.sum(ss['dΩ_tilde_dp'] * dlogp, axis = 2) + shocks['dΩ_tilde_zt'])
        
    #     dλ_Φ = dΦ @ ss['Ψ']
    #     dλ_μ = - (ss['λ'] * dlogμ) @ ss['Ω'] @ ss['Ψ']
    #     dλ_T = - ss['λ'] @ (shocks['dt'] * ss['Ω']) @ ss['Ψ']  # Elementwise multiplication
        
    #     dλ_Ω =    (ss['λ'] / ss['μ']) @ dΩ_tilde  @ ss['Ψ']
    #     dλ   = dλ_Φ +  dλ_μ + dλ_T + dλ_Ω
    #     dλ_C_test  = np.zeros(dλ.shape)
    #     dλ_C_test[:ss['C']] = dλ[:ss['C']]
        
    # else:
    
    selection   = list(range(C))
    selection_K = list(range(C + ss['N'], C+ ss['N'] + ss['K']))


    dλ_C, _, _ = compute_dλ_KORV_select(sol, shocks, ss, dual, selection)
    if ss['K'] > 0:
        dλ_K, _, _ = compute_dλ_KORV_select(sol, shocks, ss, dual, selection_K)
    else:
        dλ_K = np.zeros(ss['λ'].shape)


    
    # Print the maximum absolute difference between dλ_C_test and dλ_C
    # print("Maximum difference between dλ_C methods:", np.max(np.abs(dλ_C_test - dλ_C)))
    
    # dλ, _, _ = Blanchard_functions.compute_dλ(sol, shocks, ss)
    dlogC_c = (dλ_C[:C]/λ[:C] - dlogp[:C])
    dlogC   =  np.sum(Φ[:C] * dlogC_c)
    
    
    
    C_idx_dict = {c: np.array([i for i in range(ss['λ'].shape[0]) if ss['C_map'][i] == c]) for c in range(ss['C'])}     
    λ_dict  = {c : ss['λ'][C_idx_dict[c]] for c in range(ss['C'])}
    λ_dict  = {c : ss['λ'][C_idx_dict[c]] for c in range(ss['C'])}
    
    μ_dict  = {c : ss['μ'][C_idx_dict[c]] for c in range(ss['C'])}

    cap_by_c = range(1 + ss['N_c'], 1 + ss['N_c'] + ss['K_c'])

    if ss['K'] > 0:
        GNE_c     =  np.array([ss['Φ_c'][c] + np.sum(λ_dict[c][cap_by_c] / μ_dict[c][cap_by_c]) for c in range(ss['C'])])


        dlogK = np.where(ss['λ'] != 0, dλ_K / ss['λ'], 0) - dlogp


        dlogGNE_c = np.zeros(ss['C'])
        dlogK_c   = np.zeros(ss['C'])
        
        dlogK_c_ind = np.zeros((ss['C'], ss['K_c']))
        #dlogR_c   = np.zeros(ss['C'])
        #dr_i      = compute_dr_i(dr, dS_c, shocks, ss)
        
        for c in range(ss['C']):
            # First element is normalized consumption share
            cons_share = ss['Φ_c'][c] / GNE_c[c]
            
            # Subsequent elements are normalized capital shares
            inv_shares = (λ_dict[c][cap_by_c] / μ_dict[c][cap_by_c]) / GNE_c[c]
            cap_shares = (λ_dict[c][cap_by_c]) / GNE_c[c]

            # Combine into single array with consumption share first
            
            dlogK_c_ind[c] = dlogK[C_idx_dict[c]][cap_by_c]
            
            #dr_i_c  = dr_i[C_idx_dict[c]][cap_by_c]
            #R_i_c   = ss['r_i'][c, C_idx_dict[c]] + ss['r_i']
            
            #dlogR_c[c] =  cap_shares / np.sum(cap_shares) * dr_i_c / ()
            
            dlogGNE_c[c] =  cons_share * dlogC_c[c] + np.sum(inv_shares * dlogK_c_ind[c])
            
            dlogK_c[c] = np.sum(dlogK_c_ind[c] * cap_shares) / np.sum(cap_shares)
            
    else:
        
        dlogGNE_c = dlogC_c 
        dlogK_c  =  np.zeros(ss['C'])
        dlogK_c_ind = np.zeros((ss['C'], ss['K_c']))
        
    #print('Change in Sharpe ratio', dS_c)
    #print('Change in risk-free rate', dr)
    print('Change in country-wise real consumption is: ', dlogC_c)
    print('Change in world real consumption is: ', dlogC)
    #print('Change in NFA is ', db_c)

    # sol = {'ss' : ss, "shock" : shocks, 'dr': dr, 'dS_c': dS_c, 'sum_dλ_v1': sum_dλ_v1, 'sum_dλ_v2': sum_dλ_v2, \
    #        'sum_dλ_v3': sum_dλ_v3, 'sum_dλ_v4': sum_dλ_v4, 'sum_dλ_v5': sum_dλ_v5, 'dΛ_f':dΛ_f, 'dlogC': dlogC, 'dr_i': dr_i, \
    #         'dlogμ': dlogμ, 'dlogp': dlogp, 'dΦ': dΦ, 'dD_c': dD_c, 'db_c': db_c, 'dS_c_tilde': dS_c_tilde}
    
    
    
    
    
    sol = {'ss' : ss, "shock" : shocks, \
        'sol': {'dr': dr, 'dS_c': dS_c, 'sum_dλ_v1': sum_dλ_v1, 'sum_dλ_v2': sum_dλ_v2, \
           'sum_dλ_v3': sum_dλ_v3, 'sum_dλ_v4': sum_dλ_v4, 'sum_dλ_v5': sum_dλ_v5, 'dΛ_f':dΛ_f,  'dlogC_c': dlogC_c, 'dlogC': dlogC, 'dr_i': dr_i, \
            'dlogμ': dlogμ, 'dlogp': dlogp, 'dΦ': dΦ, 'dD_c': dD_c, 'db_c': db_c, 'dS_c_tilde': dS_c_tilde, \
            'dg_ωc': dg_ωc, 'dχ_c': dχ_c, 'dλ_K' : dλ_K, 'dlogGNE_c' : dlogGNE_c, 'dλ_C' : dλ_C, 'dλ_K' : dλ_K, 'dlogK_c' : dlogK_c, 'dlogK_c_ind' : dlogK_c_ind}}
    
    λ_tilde, Ψ_tilde, Ω_tilde, δ, r_i, N, K = \
        [ss[x] for x in ['λ_tilde', 'Ψ_tilde', 'Ω_tilde', 'δ', 'r_i', 'N', 'K']]
    dlogA_tilde, dlogz, dδ, dlogμ_tilde, dt = \
        [shocks[x] for x in ['dlogA_tilde', 'dlogz', 'dδ' , 'dlogμ_tilde', 'dt']]
    
    first_term  = np.sum(λ_tilde[C:] * dlogA_tilde[C:]) \
        - np.sum(Φ[:C] * np.sum(Ψ_tilde[:C,:] * np.sum(Ω_tilde * dlogz, axis=1)[None, :], axis=1))
    second_term = -np.sum(λ_tilde[C+N:C+N+K] * δ * dδ / (δ + g))
    third_term  = -np.sum(λ_tilde[C+N:C+N+K] * dr_i / (r_i + δ))
    fourth_term = -np.sum(λ_tilde[-F:] * dΛ_f / λ[-F:])
    fifth_term = -np.sum(λ_tilde[C:C+N] * dlogμ_tilde) 
    
    LHS = np.sum(Φ[:C] * (dλ_C[:C]/λ[:C] - dlogp[:C]))


    if np.sum(np.abs(dt)) == 0:
        RHS = first_term + second_term + third_term + fourth_term + fifth_term
        print('World Technology', first_term)
        print('World Reallocation', second_term + third_term + fourth_term + fifth_term)
    else: ### tariff shock
        RHS = - np.sum(λ_tilde * np.sum(Ω_tilde * dt, axis=1)) +  third_term + fourth_term
    
    #print('Prop 2 LHS = ', LHS)
    #print('Prop 2 RHS = ', RHS)
    
    if np.isclose(LHS, RHS):
        print("Proposition 2: " , np.isclose(LHS, RHS) )
    else:
        print("LHS:", LHS)
        print("RHS", RHS)
        print("Proposition 2 does not hold. Absolute difference: ", np.abs(LHS - RHS))
        assert False
    
    print("Done!")
    print("")
    return sol



def populate_shocks(ss, shocks):
    
    # Populate shocks with right shape
    # Populate shocks with zeros of the right shape if not provided    
    shocks['dg_L']         = shocks.get('dg_L', 0.0)
    shocks['dg_A']         = shocks.get('dg_A', 0.0)
    shocks['dg']           = shocks.get('dg', 0.0)
    shocks['dδ']           = shocks.get('dδ', np.zeros(ss['K']))
    shocks['dγ']           = shocks.get('dγ', np.zeros_like(ss['γ']))
    shocks['dρ_c']         = shocks.get('dρ_c', np.zeros(ss['C']))
    shocks['dν_c']         = shocks.get('dν_c', np.zeros(ss['C']))
    shocks['dσ']           = shocks.get('dσ', np.zeros(ss['K']))
    shocks['dψ']           = shocks.get('dψ', np.zeros(ss['K']))
    shocks['dlogA']        = shocks.get('dlogA', np.zeros_like(ss['λ']))
    shocks['dlogμ_tilde']  = shocks.get('dlogμ_tilde', np.zeros_like(ss['μ_tilde']))
    shocks['dτ_c']         = shocks.get('dτ_c', np.zeros(ss['C']))
    shocks['dlogz']        = shocks.get('dlogz', np.zeros_like(ss['Ω']))
    shocks['dt']           = shocks.get('dt'   , np.zeros_like(ss['Ω']))

    shocks['dlogA_tilde']     =  shocks['dlogA']  # DO NOT SHOCK DEPRECIATION
    
    
    ## Old 


    if 'dΩ_tilde_dp' in ss:
        shocks['dΩ_tilde_zt']  = np.einsum('ijp,ip->ij', ss['dΩ_tilde_dp'], shocks['dlogz'] + shocks['dt'])

    return shocks



#%%
def unpack_dict(d, keys):
    return [d[key] for key in keys]


def log_linearized_fun(x0, shocks, ss, full):
    '''
    x[0]: ∑(1-1/μ)dλ (shape: C x 1)
    x[1]: ∑(1-1/μ_tilde)dλ (shape: C x 1)
    x[2]: ∑r_i dλ/(r_i + δ_i) (shape: C x 1)
    x[3]: ∑(σ_iψ_i)dλ/(r_i + δ_i) (shape: C x 1)
    x[4]: ∑dλ/(r_i + δ_i) (shape: C x 1)
    x[5]: dΛ_f (shape: F x 1)
    x[6]: dS_c (shape: C x 1)
    x[7]: dr (shape: scalar)
    x[8]: db_c (shape: C x 1)
    x[9]: dD_c (shape: C x 1)
    x[10]: dS_c_tilde (shape: C x 1)
    '''
    # unpack arguments into correct shapes
    x = unpack_x(x0, ss)
    # print('x is ', x[0])
    # unpack parameters
    
    dν_c, dτ_c, dδ, dσ, dψ, dγ, dlogz, dlogA, dg, dg_L, dg_A, dt = \
        tuple(shocks.get(name, 0.0) for name in ['dν_c', 'dτ_c', 'dδ', 'dσ', 'dψ', 'dγ', \
                                              'dlogz', 'dlogA', 'dg', 'dg_L', 'dg_A', 'dt'])
        
    θ, λ, r, r_i, χ_c, μ, Ψ, Φ, Ω, Ω_tilde, S_c, S_c_tilde, ν_c, g, \
        g_A, g_L, D_c, τ_c, δ, σ, ψ, γ, t, g_ωc, C, N, K, F, K_c, F_c, N_c \
            = unpack_dict(ss, ['θ', 'λ', 'r', 'r_i', 'χ_c', 'μ', \
                                         'Ψ', 'Φ', 'Ω', 'Ω_tilde', 'S_c', 'S_c_tilde', \
                                        'ν_c', 'g', 'g_A', 'g_L', 'D_c', 'τ_c', 'δ', \
                                        'σ', 'ψ', 'γ', 't', 'g_ωc', 'C', 'N', 'K', 'F', \
                                        'K_c', 'F_c', 'N_c'])
    #########################################################
    if full:
    ### full effect of shocks
        dr_i = compute_dr_i(x[7], x[6], shocks, ss) # shape: K x 1
    ### partial effect of shocks
    else:
        dr_i = np.zeros(K) # shape: K x 1
    # dlogμ = compute_dlogμ(dr_i, shocks, ss) # shape: C+N+K+F x 1
    #########################################################
    eq1_term, eq3_term, eq4_term, eq5_term, eq6_term = compute_dλ_new(x0, shocks, ss, full)
    # print('Hannes eq1_term is: ', eq1_term)
    # print('Hannes eq3_term is: ', eq3_term)
    # print('Hannes eq4_term is: ', eq4_term)
    # print('Hannes eq5_term is: ', eq5_term)
    # print('Hannes eq6_term is: ', eq6_term)
    # dλ, _, _ = compute_dλ(x0, shocks, ss)
    # eq1_term = np.sum(np.reshape((1 - 1/μ[C+N:C+N+K]) * dλ[C+N:C+N+K], (C, K_c)), axis=1)
    # eq2_term = np.sum(np.reshape((1 - 1/μ[C:C+N]) * dλ[C:C+N], (C, N_c)), axis=1)
    # eq3_term = np.sum(np.reshape((r_i / (r_i + δ)) * dλ[C+N:C+N+K], (C, K_c)), axis=1)
    # eq4_term = x[3] - np.sum(np.reshape((σ * ψ / (r_i + δ)) * dλ[C+N:C+N+K], (C, K_c)), axis=1)
    # eq5_term = np.sum(np.reshape((1 / (r_i + δ)) * dλ[C+N:C+N+K], (C, K_c)), axis=1)
    # eq6_term = dλ[C+N+K:]
    # print('Yutong eq1_term is: ', eq1_term)
    # print('Yutong eq2_term is: ', eq2_term)
    # print('Yutong eq3_term is: ', eq3_term)
    # print('Yutong eq4_term is: ', eq4_term)
    # print('Yutong eq5_term is: ', eq5_term)
    # print('Yutong eq6_term is: ', eq6_term)

    eq2_term = 0 
   
    #dλ, _, _ = compute_dλ(x0, shocks, ss)  
    
    # ### equation 1 (∑(1-1/μ)dλ)
    eq1 = x[0]  - eq1_term
    ### equation 2 (∑(1-1/μ_tilde)dλ)
    eq2 = x[1]  - eq2_term
    ### equation 3 (∑r_i dλ/(r_i + δ_i))
    eq3 = x[2]  - eq3_term
    ### equation 4 (∑(σ_iψ_i)dλ/(r_i + δ_i))
    eq4 = x[3]  - eq4_term
    ## equation 5 (∑dλ/(r_i + δ_i))
    eq5 = x[4]  - eq5_term
    ### equation 6 (dΛ_f) ### NOTE: shape F x 1
    eq6 = x[5]  - eq6_term
    
    
   
    # dλ = eq1_term 
    # dlogp = compute_dlogp(x[5], dlogA, dlogμ, shocks, ss) # shape: C+N+K+F x 1
    dlogg_ωc = compute_dlogg_ωc(x[7], x[6], shocks, ss) # shape: C x 1
    dg_ωc = dlogg_ωc * g_ωc # shape: C x 1
    dlogχ_c = compute_dlogχ_c(dlogg_ωc, shocks, ss) # shape: C x 1
    # dΦ = compute_dΦ(dr_i, x[5], x[9], x[8], x[0], x[7], shocks, ss) # shape: C+N
    # dlogΦ = np.divide(dΦ, Φ, out=np.zeros_like(dΦ), where=Φ!=0) # shape: C+N+K+F x 1

    

    ### equation 7 (dS_c) 
    dlogσ = dσ / σ
    dlogψ = dψ / ψ
    dχ_c = dlogχ_c * χ_c
    dlogγ = dγ / γ

    eq7 = np.zeros(C)
    if full:
        for c in range(C):
            K_full  = np.array([k for k in range(C + N, C + N +K) if ss['C_map'][k] == c])
            K_K     = K_full - C - N
            
            eq7[c] = x[3][c] \
                    + np.sum(
                            σ[K_K] * ψ[K_K] \
                                * λ[K_full]/(r_i[K_K] + δ[K_K])
                            * (dlogσ[K_K] + dlogψ[K_K] \
                               - (1/(r_i[K_K] + δ[K_K])) \
                                * (dr_i[K_K] + dδ[K_K]))
                        ) \
                    - (np.sum(λ[C+N+K+c*F_c:C+N+K+(c+1)*F_c] * (1+D_c[c])) * χ_c[c] \
                       * γ * S_c[c] / ((1-τ_c[c]) * r + ν_c[c] - g_A)) \
                        * (dlogγ + (x[6][c] / S_c[c]) + (1/(ν_c[c] +g_L)) \
                           * (dν_c[c] + dg_L) \
                    - (1/(ν_c[c] + g + g_ωc[c])) * (dg + dν_c[c] - dg_ωc[c]) \
                        + (np.sum(x[5][c*F_c:(c+1)*F_c])/np.sum(λ[C+N+K+c*F_c:C+N+K+(c+1)*F_c])) \
                    - (1/((1-τ_c[c]) * r + ν_c[c] - g_A)) * (-dτ_c[c] * r + (1-τ_c[c]) \
                            * x[7] + dν_c[c] - dg_A) + x[9][c] / (1+D_c[c]))
                        
                        
            
            
    else:
        for c in range(C):
            eq7[c] = x[6][c]

    ### equation 8 (dr)
    if full == True:
        # eq8 = np.sum(x[4]) - np.sum((λ[C+N:C+N+K]/((r_i + δ) ** 2)) * (dr_i + dδ)) \
        #     - np.sum(x[10])
        eq8 = np.sum(x[8])
    else:
        eq8 = x[7]

    ### equation 9 (db_c)
    if full == True:
        eq9 = x[8] - compute_db_c(x[10], dr_i, x[4], shocks, ss)
    else:
        eq9 = x[8]

    ### equation 10 (dD_c)
    eq10 = x[9] - compute_dD_c(x[7], dr_i, x[8], x[5], x[1], x[2], shocks, ss) 

    ### equation 11 (dS_c_tilde)
    eq11 = x[10] - compute_dlogS_c_tilde(x[5], x[9], x[7], dlogχ_c, shocks, ss) * S_c_tilde

    list_eqs = np.concatenate([eq1.flatten(), eq2.flatten(), eq3.flatten(), eq4.flatten(), \
                               eq5.flatten(), eq6.flatten(), eq7.flatten(), eq8.flatten(), \
                                eq9.flatten(), eq10.flatten(), eq11.flatten()])

    return list_eqs



def linear_funcs_tax(x, ss):
    '''
    Solve for S_c_tilde (financial wealth), D_c (tax revenues) and b_c (NFA)
    x[0]: S_c_tilde (shape: C x 1)
    x[1]: D_c (shape: C x 1)
    x[2]: b_c (shape: C x 1)
    '''
    λ, r_i, r, χ_c, K_c, C, F_c, N, K, N_c, g_A, δ, ν_c, τ_c, μ_tilde \
        = unpack_dict(ss, ['λ', 'r_i', 'r', 'Χ_c', 'K_c', 'C', 'F_c', 'N', 'K', 'N_c', 'g_A', 'δ', 'ν_c', 'τ_c', 'μ_tilde'])

    ### equation on b_c (shape: C x 1)
    eq1 = x[2] - x[0] + np.sum(np.reshape(λ[C+N:C+N+K]/(r_i + δ), (C, K_c)), axis=1) 

    ### equation on S_c_tilde (shape: C x 1)
    eq2 = x[0] - np.sum(np.reshape(λ[C+N+K:], (C, F_c))*(1+x[1]), axis=1) / ((1-τ_c) * r + ν_c - g_A) * (χ_c - 1)

    ### equation on D_c (shape: C x 1)
    eq3 = x[1] - (τ_c * ((r * x[2]) + np.sum(np.reshape((λ[C+N:C+N+K] * r_i / (r_i + δ)), (C, K_c)), axis=1)) \
                   + np.sum(np.reshape(λ[C:C+N] * (1-1/μ_tilde), (C, N_c)), axis=1)) / np.sum(np.reshape(λ[C+N+K:], (C, F_c)), axis=1)
    
    list_eqs = np.concatenate([eq1.flatten(), eq2.flatten(), eq3.flatten()])
    return list_eqs
      
### Wedge shock (shape: C+N+K+F x 1)
def compute_dlogμ(dr_i, shocks, ss):
    # unpack parameters
    dlogμ_tilde, dδ, dg = tuple(shocks.get(name, 0.0) for name in ['dlogμ_tilde', 'dδ', 'dg'])
    δ, g_A, g_L, r_i, C, N, K, F = \
        unpack_dict(ss, ['δ', 'g_A', 'g_L', 'r_i', 'C', 'N', 'K', 'F'])
    g = g_A + g_L
    dlogμ = np.zeros(C+N+K+F)
    dlogμ[C:C+N] = dlogμ_tilde.copy() # tax shock on goods
    # replace the capital asset part
    dlogμ[C+N:C+N+K] += ((dr_i + dδ) / (r_i + δ)) - ((dg + dδ) / (g + δ))
    return dlogμ

def compute_dlogp(dΛ_f, dlogA, dlogμ, shocks, ss): # (shape: C+N+K+F x 1)
    # unpack
    dlogz, dt = tuple(shocks.get(name, 0.0) for name in ['dlogz', 'dt'])
    λ, Ψ_tilde, Ω_tilde, C, N, K = \
        unpack_dict(ss, ['λ', 'Ψ_tilde', 'Ω_tilde', 'C', 'N', 'K'])
    Λ_f = λ[C+N+K:]
    dΛ_f = np.float64(dΛ_f)
    dlogΛ_f = np.divide(dΛ_f, Λ_f, out=np.zeros_like(dΛ_f), where=Λ_f!=0.0)

    dlogp = Ψ_tilde @ (dlogμ - dlogA + np.sum(Ω_tilde * dlogz, axis=1) \
                       + np.sum(Ω_tilde * dt, axis=1)) \
                       + np.sum(Ψ_tilde[:,C+N+K:] * dlogΛ_f, axis=1) 
    
    return dlogp


### Interest rates shocks of different capital goods (shape: K x 1)
def compute_dr_i(dr, dS_c, shocks, ss):
    # unpack parameters
    dσ, dψ, dτ_c = tuple(shocks.get(name, 0.0) for name in ['dσ', 'dψ', 'dτ_c'])
    S_c, ψ, τ_c, σ, C, K, K_c = unpack_dict(ss, ['S_c', 'ψ', 'τ_c', 'σ', 'C', 'K', 'K_c'])
    dlogσ = np.divide(dσ, σ, out=np.zeros_like(dσ), where=σ!=0.0) # shape: K x 1
    dlogψ = np.divide(dψ, ψ, out=np.zeros_like(dψ), where=ψ!=0.0) # shape: K x 1
    dlogS_c = np.divide(np.float64(dS_c), np.float64(S_c), out=np.zeros(C), where=np.float64(S_c)!=0.0) # shape: C x 1
    dr_i     = np.zeros(K)

    ss = ss
    for k in range(K):
        i = ss['C'] + ss['N'] + k
        c = ss['C_map'][i]
        term1 = dr
        term2 = σ[k] * ψ[k] * S_c[c] / (1 - τ_c[c])
        term3 = dlogσ[k] + dlogψ[k] + dlogS_c[c] + dτ_c[c] /(1-τ_c[c])
        dr_i[k] = term1 + term2 * term3
    
    return dr_i

def compute_dlogg_ωc(dr, dS_c, shocks, ss):
    # unpack parameters
    dρ_c, dγ, dτ_c      = [shocks[x] for x in ['dρ_c','dγ', 'dτ_c']]
    S_c, g_ωc, r, γ, τ_c = [ss[x] for x in ['S_c', 'g_ωc', 'r', 'γ', 'τ_c']]
    
    dlogγ = dγ / γ # shape: scalar
    

    dlogg_ωc = dlogγ + (γ / g_ωc) \
        * (-dτ_c * r + (1-τ_c) * dr - dρ_c \
           + (dγ / 2 * ((S_c * (1-τ_c)) ** 2)) \
           - (γ + 1) * (1-τ_c) * dτ_c * (S_c ** 2) \
            + (γ + 1) * ((1-τ_c) ** 2) * (S_c * dS_c)) # shape: C x 1
    return dlogg_ωc

def compute_dlogχ_c(dlogg_ωc, shocks, ss):
    dν_c, dg_L, dg    = [shocks[x] for x in ['dν_c', 'dg_L', 'dg']]
    ν_c, g_L, g, g_ωc = [ss[x] for x in ['ν_c', 'g_L', 'g', 'g_ωc']]

    dg_ωc = g_ωc * dlogg_ωc # shape: C x 1
    dlogχ_c = (dν_c + dg_L) / (ν_c + g_L) - (dν_c + dg - dg_ωc) / (ν_c + g - g_ωc) # shape: C x 1
    return dlogχ_c

def compute_dD_c(dr, dr_i, db_c, dΛ_f, sum_dλ_v2, sum_dλ_v3, shocks, ss):
    '''
    NOTE: 
    dΛ_f (shape: F x 1)
    sum_dλ_v2 (shape: C x 1)
    sum_dλ_v3 (shape: C x 1)
    '''
    dτ_c, dδ, dlogμ_tilde, dt = tuple(shocks.get(name, 0.0) for name in ['dτ_c', 'dδ', 'dlogμ_tilde', 'dt'])

    r, r_i, λ, δ, μ_tilde, b_c, τ_c, D_c, C, N, K, F_c, K_c, N_c, Ω, t = \
        unpack_dict(ss, ['r', 'r_i', 'λ', 'δ', 'μ_tilde', 'b_c', \
                                   'τ_c', 'D_c', 'C', 'N', 'K', 'F_c', 'K_c', 'N_c', 'Ω', 't'])
    
    temp =   np.sum( ss['C_agg'] @ (np.diag(λ) @ (dt * Ω)), axis = 1)
    

        
    dD_c = (1/np.sum(np.reshape(λ[C+N+K:], (C, F_c)), axis=1)) \
        * (dτ_c * (r * b_c + np.sum(np.reshape(r_i / (r_i  + δ) * λ[C+N:C+N+K], (C, K_c)), axis=1)) \
        + τ_c * (dr * b_c + r * db_c + sum_dλ_v3 + np.sum(np.reshape(np.divide(dr_i, (dr_i + dδ),\
            out=np.zeros_like(dr_i), where=(dr_i + dδ)!=0) * λ[C+N:C+N+K], (C, K_c)), axis=1)) \
        + sum_dλ_v2 + np.sum(np.reshape((dlogμ_tilde * λ[C:C+N]) / μ_tilde, (C, N_c)), axis=1) \
        - D_c * np.sum(np.reshape(dΛ_f, (C, F_c)), axis=1) + temp)
    

    return dD_c


    
    

def compute_db_c(dS_c_tilde, dr_i, sum_dλ_v5, shocks, ss):
    '''
    NOTE: 
    dS_c_tilde (shape: C x 1)
    sum_dλ_v5 (shape: C x 1)
    '''
    r_i, λ, δ, C, N, K, K_c = unpack_dict(ss, ['r_i', 'λ', 'δ', 'C', 'N', 'K', 'K_c'])
    dδ = shocks['dδ']

    
    db_c = dS_c_tilde - sum_dλ_v5 + \
            ss['C_agg'][:, C+N:C+N+K]  @  (λ[C+N:C+N+K] / ((r_i + δ) ** 2) * (dr_i + dδ))
    # shape: C x 1
    return db_c

def compute_dlogS_c_tilde(dΛ_f, dD_c, dr, dlogχ_c, shocks, ss):
    dν_c, dg_A, dτ_c = tuple(shocks.get(name, 0.0) for name in ['dν_c', 'dg_A', 'dτ_c'])
    λ, r, χ_c, ν_c, g_A, τ_c, D_c, C, N, K, F_c = \
        unpack_dict(ss, ['λ', 'r', 'χ_c', 'ν_c', 'g_A', 'τ_c', 'D_c', 'C', 'N', 'K', 'F_c'])

    dlogS_c_tilde = np.sum(np.reshape(dΛ_f, (C, F_c)), axis=1) / np.sum(np.reshape(λ[C+N+K:], (C, F_c)), axis=1) \
        + dD_c / (1+D_c) \
            - (-dτ_c * r + (1-τ_c) * dr + dν_c - dg_A) / ((1-τ_c) * r + ν_c - g_A) \
            + (χ_c / (χ_c - 1)) * dlogχ_c # shape: C x 1
    return dlogS_c_tilde


def compute_dΦ(dr_i, dΛ_f, dD_c, db_c, sum_dλ_v1, dr, shocks, ss):
    λ, r, μ, g, D_c, b_c, C, N, K, F, K_c, F_c = unpack_dict(ss, ['λ', 'r', 'μ', 'g', 'D_c', 'b_c', 'C', 'N', 'K', 'F', 'K_c', 'F_c'])
    dg = shocks['dg']
    dlogμ = compute_dlogμ(dr_i, shocks, ss)
   
    dΦ_c = (1+D_c) * np.sum(np.reshape(dΛ_f, (C, F_c)), axis=1) \
           + np.sum(np.reshape(λ[C+N+K:], (C, F_c)), axis=1) * dD_c \
           + sum_dλ_v1 \
           + ss['C_agg'][:,C+N:C+N+K] @ (dlogμ[C+N:C+N+K] * (λ[C+N:C+N+K]/μ[C+N:C+N+K])) \
           + db_c * (r - g) \
           + b_c * (dr - dg) # shape: C x 1
    dΦ = np.zeros(C+N+K+F)
    dΦ[:C] = dΦ_c.copy()
    # Normalize so sum is zero
    return dΦ


@njit
def WeightedCov(weight, vec1, vec2): # weight is a vector
    return ((weight * vec1) * vec2).sum() - (vec1 * weight).sum() * (vec2 * weight).sum()
    # return (weight * vec1) @ vec2 - (vec1 @ weight) * (vec2 @ weight)
  
def unpack_x(x0, ss):
    C, F = unpack_dict(ss, ['C', 'F'])
    dims = [C, C, C, C, C, F, C, 1, C, C, C]
    x1 = []
    i = 0
    for dim in dims:
        x1.append(x0[i: i + dim])
        i += dim
    return x1


@njit
def Renormalization_Ω_weight(Ω, k, n, N_sectors, C, N, dual): # n indexes sector
    Ω_k = Ω[k,:]
    ### if row sum is empty (prevention)
    
    step_size = N_sectors if dual == 0 else N_sectors / 2 
    if np.sum(Ω_k[C+n:C+N:step_size]) == 0: 
        weight_new = np.zeros(C)
    else:
        weight_new = Ω_k[C+n:C+N:step_size] / np.sum(Ω_k[C+n:C+N:step_size])
    renormalized_Ω_k = np.zeros(len(Ω_k))
    renormalized_Ω_k[C+n:C+N:step_size] = weight_new.copy()
    return renormalized_Ω_k



def compute_dλ_new(sol0, shocks, ss, full):
    # unpack solution to the correct shapes
    sol = unpack_x(sol0, ss)
    # unpack parameters
    dlogz, dlogA, dt = tuple(shocks.get(name, 0.0) for name in ['dlogz', 'dlogA', 'dt'])
    eq_matrices = ss['eq_matrices']

    Psi_mats = eq_matrices['Psi_mats']
    mu_mats  = eq_matrices['mu_mats']
    t_mats   = eq_matrices['t_mats']
    
    
    if full:
        dr_i     = compute_dr_i(sol[7], sol[6], shocks, ss)
    else:
        dr_i     = np.zeros(ss['K'])
    dlogμ    = compute_dlogμ(dr_i, shocks, ss)
    dlogp    = compute_dlogp(sol[5], dlogA, dlogμ, shocks, ss)
    dΦ       = compute_dΦ(dr_i, sol[5], sol[9], sol[8], sol[0], sol[7], shocks, ss) 
    dlogΦ    = np.divide(dΦ, ss['Φ'], out=np.zeros_like(dΦ), where=ss['Φ']!=0)

    C, N, K, F, N_c, θ, Ω_tilde, t, Ω, λ, μ, Φ, Ψ, Φ_c, K_c, r_i, δ   \
        = [ss[x] for x in ['C', 'N', 'K', 'F', 'N_c', 'θ', 'Ω_tilde', 't', 'Ω', 'λ', 'μ', 'Φ', 'Ψ', 'Φ_c', "K_c", 'r_i', 'δ']]
    ψ, σ = ss['ψ'], ss['σ']
    
    dual = ss.get("dual", 0)
    
    
    #  Fast implementation 
    if 'dΩ_tilde_dp' in ss:
        dΩ_tilde_dp = ss['dΩ_tilde_dp']
        dΩ_tilde_zt = shocks['dΩ_tilde_zt']
        dΩ_norm =   (λ / μ) @ (np.sum(dΩ_tilde_dp * dlogp, axis = 2) + dΩ_tilde_zt)
        cov  =  [  (dΦ - np.sum(dΦ) * Φ  +   dΩ_norm)  @ Ψ_eq   for Ψ_eq in Psi_mats ]
    else: 
        cov  = [compute_dλ_inner_cov(C, N, K, F, N_c, θ, dt, dlogp, dlogz, dlogΦ, dlogμ, Ω_tilde, t, Ω, Ψ_eq, λ, μ, Φ, dual) for Ψ_eq in Psi_mats]  


    #cov  = [compute_dλ_inner_cov(C, N, K, F, N_c, θ, dt, dlogp, dlogz, dlogΦ, dlogμ, Ω_tilde, t, Ω, Ψ_eq, λ, μ, Φ, dual) for Ψ_eq in Psi_mats]  


    #Test equality of cov and cov_new
    # for i in range(len(cov)):
    #     assert np.allclose(cov[i], cov_new[i]), f"Covariance vectors not equal at index {i}"
    # # cov_dΩ  = np.einsum('ijk,k->ij', ss['dΩ_tilde_dp'], dlogp)
    # # dΨ = Ψ @ cov_dΩ @ Ψ  # Shape: (N,N)
    # # cov_dλ =  Φ @ dΨ  # Shape: (N,)
    # #cov = [cov_dλ @ Ψ_eq  + dΦ @  Ψ_eq for Ψ_eq in Psi_mats]
           
    mu_terms  =  [-(λ * dlogμ) @ mu_mat for mu_mat in mu_mats]
    t_terms   =  [-λ @ (dt * Ω) @ t_mat for t_mat in t_mats]
    
    return [cov[i] + mu_terms[i] + t_terms[i] for i in range(len(cov))]
    


@njit
def compute_dλ_inner_cov(C, N, K, F, N_c, θ, dt, dlogp, dlogz, dlogΦ, dlogμ, Ω_tilde, t, Ω, Ψ_eq, λ, μ, Φ, dual):
    cov     = np.zeros(Ψ_eq.shape[1])
    temp1 = np.zeros(Ψ_eq.shape[1])
    for i in range(Ψ_eq.shape[1]):
        # Original weighted covariance calculation
        temp1[i] = WeightedCov(Φ, Ψ_eq[:,i], dlogΦ)
        
        #temp1[i] = ((Φ * Ψ_eq[:,i]) * dlogΦ).sum() - (Ψ_eq[:,i] * Φ).sum() * (Φ).sum()
        for k in range(C + N + K):
            N_c_raw = N_c if dual == 0 else N_c // 2
            for n in range(N_c_raw):
                # if k >= C:
                #     c = np.floor((k-C) / N_c)
                # else:
                #     c = 0
                c = 0 # Only works with uniform theta. 
                index = np.int64(C+c*N_c+n)
                cov[i] += λ[k]/μ[k] * np.sum(Ω_tilde[k, C+n:C+N:N_c_raw])   * (1-θ[index]) \
                    * WeightedCov(Renormalization_Ω_weight(Ω_tilde, k, n, N_c, C, N, dual), \
                                  Ψ_eq[:,i], dlogp + dlogz[k,:] + dt[k,:])
    # print('temp1 is:', temp1)     
    
    return cov + temp1





def compute_dλ_KORV_select(sol0, shocks, ss, dual, selection):
    # unpack solution to the correct shapes
    sol = unpack_x(sol0, ss)
    # unpack parameters
    dlogz, dlogA, dt = tuple(shocks.get(name, 0.0) for name in ['dlogz', 'dlogA', 'dt'])
    λ, μ, Ψ, Ω, Ω_tilde, Φ, θ, C, N, K, F, N_c = \
        unpack_dict(ss, \
                    ['λ', 'μ', 'Ψ', 'Ω', 'Ω_tilde', 'Φ', 'θ', 'C', 'N', 'K', 'F', 'N_c'])

    dr_i = compute_dr_i(sol[7], sol[6], shocks, ss)
    dlogμ = compute_dlogμ(dr_i, shocks, ss)
   
    dlogp = compute_dlogp(sol[5], dlogA, dlogμ, shocks, ss)
    dΦ = compute_dΦ(dr_i, sol[5], sol[9], sol[8], sol[0], sol[7], shocks, ss) 
    dlogΦ = np.divide(dΦ, Φ, out=np.zeros_like(dΦ), where=Φ!=0)

    dλ = np.zeros(C+N+K+F)
    
    temp, cov, temp1 = compute_dλ_inner_select(C, N, K, F, N_c, dlogΦ, \
                                                 dlogμ, dlogp, dlogz, dt, Φ, Ψ, Ω, Ω_tilde, λ, μ, θ, dual, selection)
    dλ = temp + cov + temp1
    return dλ, dlogp, dr_i


@njit
def compute_dλ_inner_select(C, N, K, F, N_c, dlogΦ, dlogμ, dlogp, dlogz, \
                          dt, Φ, Ψ, Ω, Ω_tilde, λ, μ, θ, dual, selection):
    cov = np.zeros(C+N+K+F)
    temp = np.zeros(C+N+K+F)
    temp1 = np.zeros(C+N+K+F)
    for i in selection:
        temp1[i] = WeightedCov(Φ, Ψ[:,i], dlogΦ)
        for j in range(C, C+N+K):
            temp[i] -= λ[j] * (Ψ[j,i] - (i==j)) * dlogμ[j]
        ### tariff 
        for j_prime in range(C+N+K):
            temp[i] -= λ[j_prime] * np.sum(dt[j_prime,:] * Ω[j_prime,:] * Ψ[:,i]) 
        for k in range(C+N):
            for n in range(N_c):
                if k >= C:
                    c = np.floor((k-C) / N_c)
                else:
                    c = 0
                index = np.int64(C+c*N_c+n)
                cov[i] += λ[k]/μ[k] * np.sum(Ω_tilde[k, C+n:C+N:N_c]) * (1-θ[index]) \
                    * WeightedCov(Renormalization_Ω_weight(Ω_tilde, k, n, N_c, C, N, dual), \
                                  Ψ[:,i], dlogp + dlogz[k,:] + dt[k,:]) 
    return temp, cov, temp1


def trade_shock(C, N, K, F, N_c, country_index):
    dlogz = np.zeros((C+N+K+F, C+N+K+F))
    for c in country_index:
            dlogz[c,C:C+N] = 1
            dlogz[C+c*N_c:C+(c+1)*N_c,C:C+N] = 1
            dlogz[C+N+c*N_c:C+N+(c+1)*N_c,C:C+N] = 1
            dlogz[c, C+c*N_c:C+(c+1)*N_c] = 0
            dlogz[C+c*N_c:C+(c+1)*N_c, C+c*N_c:C+(c+1)*N_c] = 0
            dlogz[C+N+c*N_c:C+N+(c+1)*N_c, C+c*N_c:C+(c+1)*N_c] = 0
    return dlogz

def trade_export_shock(C, N, K, F, N_c, country_index, impose_country_index):
    dlogz = np.zeros((C+N+K+F, C+N+K+F))

    # remaining_country = list(set(list(np.arange(C))).difference(country_index))
    remaining_country = [impose_country_index] \
        if isinstance(impose_country_index, (int, float)) else impose_country_index
    
    country_index = [country_index] \
        if isinstance(country_index, (int, float)) else country_index
    # if np.isscalar(impose_country_index):
    #     remaining_country = [impose_country_index]
    # else:
    #     remaining_country = impose_country_index.copy()

    for k in remaining_country:
        for c in country_index:
            #### consumption rows
            dlogz[k,C+c*N_c:C+(c+1)*N_c] = 1
            #### regular goods rows
            dlogz[C+k*N_c:C+(k+1)*N_c,C+c*N_c:C+(c+1)*N_c] = 1
            #### investment goods rows
            dlogz[C+N+k*N_c:C+N+(k+1)*N_c,C+c*N_c:C+(c+1)*N_c] = 1
    return dlogz

def trade_shock_bilateral(C, N, K, F, N_c, country_index, impose_country_index):
    dlogz1 = trade_export_shock(C, N, K, F, N_c, country_index, impose_country_index)
    dlogz2 = trade_export_shock(C, N, K, F, N_c, impose_country_index, country_index)
    dlogz = dlogz1 + dlogz2
    return dlogz


###################################################################
# functions used by KORV only
def KORV_Omega_tilde(Ω_tilde, C, N, F, K, N_c, F_c): # shape: (C + N*4 + K + F) x (C + N*4 + K + F)
    new_Omega_tilde = np.zeros((C+N*4+K+F, C+N*4+K+F))
    # copy consumption expenditure on each sector 
    new_Omega_tilde[:C, C:C+N*4:4] = Ω_tilde[:C, C:C+N].copy()

    # copy sector expenditure on regular goods
    new_Omega_tilde[C:C+N*4:4, C:C+N*4:4] = Ω_tilde[C:C+N, C:C+N].copy()

    # copy sector expenditure on capital and primary factor to the value added part of the sector
    for i in range(C, C+N*4, 4):
        idx = np.int64((i-C) / 4)
        new_Omega_tilde[i, C+1+4*idx] = np.sum(Ω_tilde[C+idx, C+N:])

    # construct the HK and LM part of each country-sector pair
    for i in range(C+1, C+N*4+1, 4):
        idx = np.int64((i-C-1) / 4)
        LM_share = (np.sum(Ω_tilde[C+idx, C+N+K+1::F_c]) + np.sum(Ω_tilde[C+idx, C+N+K+2::F_c])) / np.sum(Ω_tilde[C+idx, C+N:]) # order of labor in Ω_tilde: HS, MS, LS
        HK_share = 1 - LM_share
        new_Omega_tilde[i, C+2+4*idx] = HK_share
        new_Omega_tilde[i, C+3+4*idx] = LM_share
    
    # constrct HK bundles from primary factors and investment goods
    for i in range(C+2, C+N*4+2, 4):
        idx = np.int64((i-C-2) / 4)
        c = np.int64((i-C-2) / (N_c*4))
        capital = np.sum(Ω_tilde[C+idx, C+N:C+N+K])
        high_skill = np.sum(Ω_tilde[C+idx, C+N+K::F_c])
        new_Omega_tilde[i, C+N*4+idx] = capital / (capital+high_skill)
        new_Omega_tilde[i, C+N*4+K+c*F_c] = high_skill / (capital+high_skill)
        
    # constrct LM bundles from primary factors 
    for i in range(C+3, C+N*4+3, 4):
        idx = np.int64((i-C-3) / 4)
        c = np.int64((i-C-3) / (N_c*4))
        medium_skill = np.sum(Ω_tilde[C+idx, C+N+K+1::3])
        low_skill = np.sum(Ω_tilde[C+idx, C+N+K+2::3])
        new_Omega_tilde[i, C+N*4+K+c*F_c+1] = medium_skill / (medium_skill+low_skill)
        new_Omega_tilde[i, C+N*4+K+c*F_c+2] = low_skill / (medium_skill+low_skill)

    # copy capital goods expenditure on regular goods
    new_Omega_tilde[C+N*4: C+N*4+K, C:C+N*4:4] = Ω_tilde[C+N:C+N+K, C:C+N].copy()
    return new_Omega_tilde

