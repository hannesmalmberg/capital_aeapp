

import numpy as np


def tariff(ss, orig = None, dest = None, scale = 1.0, bilateral = False):  # This is for an import tariff
    
    static = ss.get('static', False)
    # Country key  Key = C x (C + N + K + F) matrix saying with Key_{ij}=1 if  j pertains to i
    K = ss['K'] if not static else 0
    C, N, F = ss['C'], ss['N'], ss['F']
    # Get country indices from country names
    cs = list(ss['countries'])
    
    orig_idx = np.array([cs.index(iso)  for iso in orig]) if orig is not None else np.arange(C)
    dest_idx = np.array([cs.index(iso) for iso in dest]) if dest is not None else np.arange(C)

    if orig is not None and not all(c in cs for c in orig):
        raise ValueError(f"All export countries must be in: {cs}")
    if dest is not None and not all(c in cs for c in dest):
        raise ValueError(f"All import countries must be in: {cs}")
    
    dt = np.zeros((C+N+K+F, C+N+K+F)) # dt

    for i in np.arange(C + N + K):
        for j in np.arange(C + N + K):
            if ss['C_map'][i] in dest_idx and ss['C_map'][j] in orig_idx and ss['C_map'][i] != ss['C_map'][j]:
                dt[i , j] = 1.0   
                
    
    if bilateral:
        dt_rev = tariff(ss, orig = dest, dest = orig, scale = 1.0, bilateral = False)
        return scale * (dt + dt_rev)
    
    return dt * scale 



def markups(ss):  # Model markup as ingoing tariff
    dμ                = np.zeros_like(ss['Ω'])
    dμ[ss['C']:ss['C'] + ss['N'], :] = np.ones(ss['Ω'].shape[0])
    return dμ
