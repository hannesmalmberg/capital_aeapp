import numpy as np
import solution
import utils 
from scipy.integrate import trapezoid
import calibration
from IPython.utils.capture import capture_output


def solve(ss_init, shocks, iter = 10, vars = ['dlogC', 'dlogC_c'], full =  True):
    ss_iter = ss_init.copy()
        
    dvars = {}
    grid       = power_law_times(1, iter + 1, p = 2)
    dgrid      = grid[1:] - grid[:-1]
    
    
    for i in range(iter):
        print(i)
        with capture_output() as captured:
            sol        = solution.solve(ss_iter, shocks, full)
            
        for var in vars:
            if i == 0:
                dvars[var] = np.zeros((iter,) + sol['sol'][var].shape)
            
            dvars[var][i] = sol['sol'][var]
        
        ss_iter    = update_ss(ss_iter, shocks, sol['sol'], dgrid[i])
   
    ss_final = ss_iter

    for var in vars:
        dvars_tot = {var: utils.integrate_derivative_irregular(dvars[var], dgrid) for var in vars}
    
    print("Change in world consumption: ", dvars_tot['dlogC'])
    print("Change in country consumption: ",dvars_tot['dlogC_c'])
    
    return {'dvars_tot': dvars_tot, 'dvars' : dvars, 'ss_final' : ss_final}


def power_law_times(t, N, p=2.0):
    """
    Returns an array of length N from 0 to t, 
    with spacing that gets larger as i grows (p>1).
    """
    i = np.arange(N)  # 0, 1, 2, ..., N-1
    return t * (i / (N - 1))**p




def update_ss(ss, shocks, sol, step = 0.1):
    
    
    ss_new = ss.copy()
    C, N, static =  [ss[x] for x in ['C', 'N','static']]
    K = ss.get('K', 0)
    
    if not static:
        
        names = ['r_i', 'b_c', 'r', 'S_c', 'S_c_tilde', 'g_ωc', 'χ_c']
        for name in names:
            ss_new[name] = ss[name] + step * sol[f'd{name}']
        
        ss_new['T_c']        = (ss_new['r'] - ss['g']) * ss_new['b_c']
        ss_new['μ']          = np.ones(len(ss['Ω_tilde']))
        ss_new['μ'][C+N:C+N+K] = (ss_new['r_i'] + ss_new['δ']) / (ss_new['g'] + ss_new['δ'])
    ### W matrix is constructed such that ∑_o' W(jo, ko') = 1 for all jo in C, N, and K
    
    
    ss_new['Φ']       = ss['Φ'] + step * sol['dΦ']
    ss_new['Φ_c']     = ss_new['Φ'][:C]
    ss_new['D_c']     = ss['D_c'] + step * sol['dD_c']
    
    ss_new['Ω_tilde']  = update_omega_tilde(ss, shocks, sol, step)
    ss_new['Ω']        = ss_new['Ω_tilde'] / ss_new['μ'][:,None]

    ss_new['Ψ']       = utils.leontief_inv(ss_new['Ω'])
    ss_new['Ψ_tilde'] = utils.leontief_inv(ss_new['Ω_tilde'])
   
    ss_new['λ']       = ss_new['Φ'] @ ss_new['Ψ']
    ss_new['λ_tilde'] = ss_new['Φ'] @ ss_new['Ψ_tilde']


    C, N, F = ss['C'], ss['N'], ss['F']

    a_eq1      = np.diag(1 - 1/ss_new['μ'])
    a_eq3      = np.diag(utils.extend_capital(ss_new['r_i']/(ss_new['r_i'] + ss_new['δ']), C, N, K, F))    
    a_eq4      = np.diag(utils.extend_capital((ss_new['σ'] * ss_new['ψ'] / (ss_new['r_i'] + ss_new['δ'])), ss_new['C'], ss_new['N'], ss_new['K'], ss_new['F']))
    a_eq5      = np.diag(utils.extend_capital(1 / (ss_new['r_i'] + ss_new['δ']), ss_new['C'], ss_new['N'], ss_new['K'], ss_new['F']))     
    
    a_eq6      = np.eye(C + N + K + F)
    a_eq6      = a_eq6[:, C+N+K:]
    
    a_list     = [a_eq1, a_eq3, a_eq4, a_eq5]

    Psi_mats    = [ss_new['Ψ'] @ a @ ss_new['C_agg'].T for a in a_list]
    Psi_mats.append(ss_new['Ψ'] @ a_eq6)

    mu_mats    =  [(ss_new['Ψ'] - np.eye(len(ss_new['Ψ']))) @  a @ ss_new['C_agg'].T for a in a_list]
    mu_mats.append((ss_new['Ψ'] - np.eye(len(ss_new['Ψ']))) @ a_eq6)

    t_mats      = [ss_new['Ψ'] @ a @ ss_new['C_agg'].T for a in a_list]
    t_mats.append(ss_new['Ψ']  @ a_eq6)

    ss_new['eq_matrices'] = {"Psi_mats" : Psi_mats, "mu_mats" : mu_mats, "t_mats" : t_mats}
    
    
    if 'dΩ_tilde_dp' in ss:
        Ω_tilde, N_c, K_c, F_c, θ, C_map = ss_new['Ω_tilde'], ss_new['N_c'], ss_new['K_c'], ss_new['F_c'], ss_new['θ'], ss_new['C_map']
        dΩ_tilde_dp  =  calibration.make_dOmega_tilde_dp_fast(C, N, K, F, Ω_tilde, N_c, K_c, F_c, θ, C_map)
        ss_new['dΩ_tilde_dp'] = dΩ_tilde_dp    
    
    del ss
    
    return ss_new



def update_omega_tilde(ss, shocks, sol, step, new = True):
    
    
    if 'dΩ_tilde_dp' in ss and new:
        dlogp = sol['dlogp']
        
        dΩ_tilde_zt  = np.einsum('ijp,ip->ij', ss['dΩ_tilde_dp'], shocks['dlogz'] + shocks['dt'])
        dΩ_tilde     = np.sum(ss['dΩ_tilde_dp'] * dlogp, axis = 2) + dΩ_tilde_zt
        return  ss['Ω_tilde'] + step * dΩ_tilde 
    
    
    
    C, N, N_c, Ω_tilde, θ = [ss[x] for x in ['C', 'N', 'N_c', 'Ω_tilde', 'θ']]
    K = ss.get('K', 0)
    N_nest = N_c // 2 if ss.get('dual', False) else N_c
    nest_idx = lambda k: slice(C+k, C+N, N_nest)
    
    dlogz, dt = tuple(shocks.get(name, 0.0) for name in ['dlogz', 'dt'])
    dlogp = sol['dlogp']

    ### W matrix is constructed such that ∑_o' W(jo, ko') = 1 for all jo in C, N, and K
    W = np.zeros(Ω_tilde.shape)
    for j in range(len(W)):
        for nest in range(N_nest):
            W_nest = np.sum(Ω_tilde[j, nest_idx(nest)])
            if W_nest != 0:
                W[j,nest_idx(nest)] = Ω_tilde[j, nest_idx(nest)] / W_nest

    ### update Ω, Ω_tilde
    dΩ_tilde = np.zeros(Ω_tilde.shape)
    dW = np.zeros(Ω_tilde.shape)
    for ic in range(C+N+K):
        for nest in range(N_nest):
            ### get the industry k 
            dlogp_nest = dlogp[nest_idx(nest)] + dt[ic, nest_idx(nest)] + dlogz[ic,nest_idx(nest)]
            weighted_dlogp = np.sum(W[ic, nest_idx(nest)] * dlogp_nest)
            dW[ic, nest_idx(nest)] = (1 - θ[ic]) * W[ic, nest_idx(nest)] * (dlogp_nest - weighted_dlogp)
    
    dΩ_tilde = np.zeros_like(Ω_tilde)
    for j in range(C+N+K):
        for nest in range(N_nest):
            W_nest = np.sum(Ω_tilde[j,nest_idx(nest)])
            dΩ_tilde[j, nest_idx(nest)] = dW[j, nest_idx(nest)] * W_nest
            
    return Ω_tilde + step * dΩ_tilde


