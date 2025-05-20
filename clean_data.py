import numpy as np
import pandas as pd
import scipy.io
import pickle
import utils

def make_intermediate_pkls(Ns = 27,Nr = 41):
    
    
    path = "@Import/"
    try:
        CapitalData = pd.read_stata(path+'data_raw/Ding_data/WIOT_MK_97_clean_41c_v01.dta')
        Description = pd.read_stata(path+'data_raw/Ding_data/WIOT_MK_97_finalcode_desc.dta')
        CapitalData = CapitalData.sort_values(by=['Country_out', 'finalcode_out', \
                                                  'Country_in', 'finalcode_in'])
        # sorted by n --> j --> i --> k
        # n (country out)
        # j (finalcode out)
        # i (country in)
        # k (finalcode in)

    except FileNotFoundError:
        print(f"File not found: {'@Import@Import/data_raw/Ding_data/WIOT_MK_97_clean_41c_v01.dta'}")
    except IOError:
        print(f"Error reading file: {'@Import/data_raw/Ding_data/WIOT_MK_97_clean_41c_v01.dta'}")

    # value added (sub-parts)
    '''
    Baqaee & Farhi (2023) aggregate 35 sectors to 30 sectors
    Ding (2023) aggregates 35 sectors to 27 sectors
    '''

    # convert pandas to np arrays
    finalcode = np.asarray(Description['finalcode'].unique()) # 27 unique sectors

    ### Sort sectors and countries in a way consistent with WIOT data
    finalcode_order   = utils.get_sectors()
    countrycode_order = utils.get_countries()

    # # Define the custom order mappings for each column
    custom_order_sector  = {final_code : i + 1 for i, final_code in enumerate(finalcode_order)}
    custom_order_country = {country : i + 1 for i, country in enumerate(countrycode_order)}
    
    # Sort the DataFrame based on the custom order mappings
    
    col_names  = ['Country_out', 'finalcode_out', 'Country_in', 'finalcode_in']
    order_list = [custom_order_country, custom_order_sector, custom_order_country, custom_order_sector]
    key_dict   = {col_names[i] : order_list[i] for i in range(len(col_names))}
    
    CapitalData_sorted = CapitalData.sort_values(by = col_names, key=lambda x: x.map(key_dict.get(x.name)))


    #### Construct the investment flows table [n, i, j, k]
    investment_flows = np.array(CapitalData_sorted.loc[(CapitalData_sorted['finalcode_in'].isin(finalcode)) & (CapitalData_sorted['finalcode_out'].isin(finalcode)), 'iflow_njik'])

    inv_flows_njik = np.reshape(investment_flows, (Nr, Ns, Nr, Ns)) # 4D: [n, j, i, k]
    inv_flows_nk   = np.reshape(np.transpose(inv_flows_njik, (2, 1, 0, 3)), (Nr*Ns, Nr*Ns)) 

    ## NOTE: Ding's data np.sum(inv_flows_njik, axis=(1,2)) is the same as WIOT data np.sum(np.reshape(GFCF, (41, 27, 41)), axis=(0)).T
    
    ### Import WIOT 1997 data
    try:
        WIOT1997 = scipy.io.loadmat(path + "data_intermediate/WIOT1997.mat")
       
        VA_original = WIOT1997['VA'].flatten() # 1107 x 1
        grossout    = WIOT1997['grossout'].flatten() # output 1107 x 1
        IO          = WIOT1997['IO'].T # 1107 x 1107 ### transpose to be purchasing x producing matrix
        consumption_original = WIOT1997['consumption'] # 1107 x 41
        GFCF        = WIOT1997['GFCF'] # 1107 x 41
        inventory_change = WIOT1997['Inventory'] # 1107 x 41
       
        ## adjustment to value added such that row sums and column sums are the same (they are not the same because of international transport margins)
        VA = grossout - np.sum(IO, axis=1)
        VA_adjust_ratio = np.divide(VA, VA_original, out=np.zeros_like(VA), where=VA_original!=0)
        ### adjustment to consumption such that it includes inventory change
        consumption = consumption_original + inventory_change


    except FileNotFoundError:
        print(f"File not found: {'WIOT1997.mat'}")
    except IOError:
        print(f"Error reading file: {'WIOT1997.mat'}")

    ### Import WIOT 1995-2009 data (for time smoothing)
    try:
        WIOT95_09 = scipy.io.loadmat(path + 'data_intermediate/wiot/WIOT95_09.mat')
        VA_95_09_original = WIOT95_09['VA'] # 15 x 1107 
        grossout_95_09 = WIOT95_09['grossout'] # 15 x 1107 
        IO_95_09 = np.transpose(WIOT95_09['IO'], (0, 2, 1)) # 15 x 1107 x 1107 ### transpose to be purchasing x producing matrix
        consumption_95_09_original = WIOT95_09['consumption'] # 15 x 1107 x 41
        GFCF_95_09 = WIOT95_09['GFCF'] # 15 x 1107 x 41
        inventory_change_95_09 = WIOT95_09['Inventory'] # 15 x 1107 x 41
        ### adjustment to value added such that row sums and column sums are the same (they are not the same because of international transport margins)
        VA_95_09 = grossout_95_09 - np.sum(IO_95_09, axis=2)

        VA_adjust_ratio_95_09 = np.divide(VA_95_09, VA_95_09_original, out=np.zeros_like(VA_95_09), where=VA_95_09_original!=0) # 15 x 1107
        ### adjustment to consumption such that it includes inventory change
        consumption_95_09 = consumption_95_09_original + inventory_change_95_09

    except FileNotFoundError:
        print(f"File not found: {'WIOT95_09.mat'}")
    except IOError:
        print(f"Error reading file: {'WIOT95_09.mat'}")
    #### Sanity check
    # np.sum(GFCF.reshape(41, 27, 41), axis=0).T ## Suppose to equal to iflow_njik summing over ij
    inv_flows_njik_ratio = np.divide(inv_flows_njik, np.sum(np.reshape(GFCF, (41, 27, 41)), axis=0).T[:, None, None, :], out=np.zeros_like(inv_flows_njik), where=np.sum(np.reshape(GFCF, (41, 27, 41)), axis=0).T[:, None, None, :]!=0)

    ### Import WIOT SEA data
    try:
        SEA1997 = scipy.io.loadmat(path + 'data_intermediate/SEA1997.mat')
        HS_labor = (SEA1997['HS_labor'] * VA_adjust_ratio[:40*27]).reshape(40, 27) # 1080 x 1
        MS_labor = (SEA1997['MS_labor'] * VA_adjust_ratio[:40*27]).reshape(40, 27) # 1080 x 1
        LS_labor = (SEA1997['LS_labor'] * VA_adjust_ratio[:40*27]).reshape(40, 27) # 1080 x 1
        tot_labor = (SEA1997['tot_labor'] * VA_adjust_ratio[:40*27]).reshape(40, 27) # 1080 x 1
        grossout_national = SEA1997['grossout_national'].reshape(40, 27) # 1080 x 1
        capital_comp = (SEA1997['capital_comp'] * VA_adjust_ratio[:40*27]).reshape(40, 27) 
        average_gos_GFCF_ratio = SEA1997['average_gos_GFCF_ratio'].reshape(40, 27)  # 1080 x 1


    except FileNotFoundError:
        print(f"File not found: {'SEA1997.mat'}")
    except IOError:
        print(f"Error reading file: {'SEA1997.mat'}")


    ### Import SEA 1995-2009 data (for time smoothing)
    try:
        SEA95_09 = scipy.io.loadmat(path + 'data_intermediate/SEA95_09.mat')
        HS_labor95_09 = (SEA95_09['HS_labor_95_09'] * VA_adjust_ratio_95_09[:,:40*27].T).reshape(40, 27, 15) # 1080 x 15
        MS_labor95_09 = (SEA95_09['MS_labor_95_09'] * VA_adjust_ratio_95_09[:,:40*27].T).reshape(40, 27, 15) # 1080 x 15
        LS_labor95_09 = (SEA95_09['LS_labor_95_09'] * VA_adjust_ratio_95_09[:,:40*27].T).reshape(40, 27, 15) # 1080 x 15
        tot_labor95_09 = (SEA95_09['tot_labor_95_09'] * VA_adjust_ratio_95_09[:,:40*27].T).reshape(40, 27, 15) # 1080 x 15
        grossout_national95_09 = np.reshape(SEA95_09['grossout_national_95_09'].T, (15, 40, 27)) # 1080 x 15
        capital_comp95_09 = (SEA95_09['capital_comp_95_09'] * VA_adjust_ratio_95_09[:,:40*27].T).reshape(40, 27, 15) # 1080 x 15

    except FileNotFoundError:
        print(f"File not found: {'SEA95_09.mat'}")
    except IOError:
        print(f"Error reading file: {'SEA95_09.mat'}")


    ### Compute implicit currency exchange using gross output data and harmonize SEA data
    grossout_exclude_ROW = grossout[:1080].reshape(40, 27)
    xr = np.divide(grossout_national, grossout_exclude_ROW, out=np.zeros_like(grossout_national), where=grossout_exclude_ROW!=0) # shape: 40 x 27

    ### Sanity check: all sectors in a country should have the same exchange rate
    [np.corrcoef(grossout_exclude_ROW[i,:], grossout_national[i,:])[0,1] for i in range(len(grossout_exclude_ROW))]
    # np.diag(np.corrcoef(grossout_exclude_ROW, grossout_national)[:40, 40:])

    xr_per_country = np.mean(xr, axis=1)

    grossout_exclude_ROW_95_09 = grossout_95_09[:,:1080].reshape(15, 40, 27)
    xr_95_09 = np.divide(grossout_national95_09, grossout_exclude_ROW_95_09, out=np.zeros_like(grossout_national95_09), where=grossout_exclude_ROW_95_09!=0) # shape: 15 x 40 x 27
    xr_per_country_95_09 = np.mean(xr_95_09, axis=2).T # shape: 40 x 15

    ### Harmonize labor compensation across countries
    tot_compensation = np.sum(consumption, axis=0)
    consumption_weight = tot_compensation / np.sum(tot_compensation)

    HS_labor_hc = HS_labor / xr_per_country[:, None]
    MS_labor_hc = MS_labor / xr_per_country[:, None]
    LS_labor_hc = LS_labor / xr_per_country[:, None]
    tot_labor_hc = tot_labor / xr_per_country[:, None]
    tot_labor_hc[np.isnan(tot_labor_hc)] = 0

    tot_capital_hc = capital_comp / xr_per_country[:, None]
    tot_capital_hc[np.isnan(tot_capital_hc)] = 0

    temp_weight = tot_compensation[:-1] / np.sum(tot_compensation[:-1])
    ROW_HS_labor = np.sum(temp_weight[:, None] * HS_labor_hc, axis=0) # 27 x 1
    ROW_MS_labor = np.sum(temp_weight[:, None] * MS_labor_hc, axis=0) # 27 x 1
    ROW_LS_labor = np.sum(temp_weight[:, None] * LS_labor_hc, axis=0) # 27 x 1
    ROW_tot_labor = ROW_HS_labor + ROW_MS_labor + ROW_LS_labor # 27 x 1

    ROW_tot_capital = np.sum(temp_weight[:, None] * tot_capital_hc, axis = 0) # 27 x 1

    ### Append ROW labor data with all other countries
    HS_labor_C = np.append(HS_labor_hc, ROW_HS_labor)
    MS_labor_C = np.append(MS_labor_hc, ROW_MS_labor)
    LS_labor_C = np.append(LS_labor_hc, ROW_LS_labor)
    tot_labor_C = np.append(tot_labor_hc, ROW_tot_labor)
    tot_capital_C = np.append(tot_capital_hc, ROW_tot_capital)
    ### Split value added to labor and capital
    labor_weight_C = np.divide(tot_labor_C, (tot_labor_C + tot_capital_C), out=np.zeros_like(tot_labor_C), where=(tot_labor_C + tot_capital_C)!=0)
    tot_labor_C = labor_weight_C * VA # 1107 x 1
    tot_capital_C = (1-labor_weight_C) * VA # 1107 x 1
    ### Split labor to different types of labor
    HS_labor_C = tot_labor_C * np.divide(HS_labor_C, (HS_labor_C + MS_labor_C + LS_labor_C), out=np.zeros_like(HS_labor_C), where=(HS_labor_C + MS_labor_C + LS_labor_C)!=0)
    MS_labor_C = tot_labor_C * np.divide(MS_labor_C, (HS_labor_C + MS_labor_C + LS_labor_C), out=np.zeros_like(MS_labor_C), where=(HS_labor_C + MS_labor_C + LS_labor_C)!=0)
    LS_labor_C = tot_labor_C * np.divide(LS_labor_C, (HS_labor_C + MS_labor_C + LS_labor_C), out=np.zeros_like(LS_labor_C), where=(HS_labor_C + MS_labor_C + LS_labor_C)!=0)

    ### Harmonize labor compensation across countries
    tot_compensation_95_09 = np.sum(consumption_95_09, axis=1) # shape: 15 x 41
    consumption_weight_95_09 = tot_compensation_95_09 / np.sum(tot_compensation_95_09)

    HS_labor_hc95_09 = HS_labor95_09 / xr_per_country_95_09[:, None, :] # shape: 40, 27, 5
    MS_labor_hc95_09 = MS_labor95_09 / xr_per_country_95_09[:, None, :] # shape: 40, 27, 5
    LS_labor_hc95_09 = LS_labor95_09 / xr_per_country_95_09[:, None, :] # shape: 40, 27, 5
    tot_labor_hc95_09 = tot_labor95_09 / xr_per_country_95_09[:, None, :] # shape: 40, 27, 5
    tot_labor_hc95_09[np.isnan(tot_labor_hc95_09)] = 0
    tot_capital_hc95_09 = capital_comp95_09 / xr_per_country_95_09[:, None, :] # shape: 40, 27, 5
    tot_capital_hc95_09[np.isnan(tot_capital_hc95_09)] = 0 # shape: 40, 27, 5

    temp_weight95_09 = tot_compensation_95_09[:,:-1] / np.sum(tot_compensation_95_09[:,:-1], axis=1)[:, None] # shape: 15 x 40
    ROW_HS_labor95_09 = np.sum(temp_weight95_09.T[:, None, :] * HS_labor_hc95_09, axis=0) # 27 x 15
    ROW_MS_labor95_09 = np.sum(temp_weight95_09.T[:, None, :] * MS_labor_hc95_09, axis=0) # 27 x 15
    ROW_LS_labor95_09 = np.sum(temp_weight95_09.T[:, None, :] * LS_labor_hc95_09, axis=0) # 27 x 15
    ROW_tot_labor95_09 = ROW_HS_labor95_09 + ROW_MS_labor95_09 + ROW_LS_labor95_09 # 27 x 15
    ROW_tot_capital95_09 = np.sum(temp_weight95_09.T[:, None,:] * tot_capital_hc95_09, axis = 0) # 27 x 15

    #Witth all other countries
    HS_labor_C_95_09 = np.reshape(np.append(HS_labor_hc95_09, ROW_HS_labor95_09[None,:,:], axis=0), (1107, 15)) # shape: 41 x 27 x 15
    MS_labor_C_95_09 = np.reshape(np.append(MS_labor_hc95_09, ROW_MS_labor95_09[None,:,:], axis=0), (1107, 15)) # shape: 41 x 27 x 15
    LS_labor_C_95_09 = np.reshape(np.append(LS_labor_hc95_09, ROW_LS_labor95_09[None,:,:], axis=0), (1107, 15)) # shape: 41 x 27 x 15
    tot_labor_C_95_09 = np.reshape(np.append(tot_labor_hc95_09, ROW_tot_labor95_09[None,:,:], axis=0), (1107, 15)) # shape: 41 x 27 x 15
    tot_capital_C_95_09 = np.reshape(np.append(tot_capital_hc95_09, ROW_tot_capital95_09[None,:,:], axis=0), (1107, 15))

    ### Split value added to labor and capital
    labor_weight_C_95_09 = np.divide(tot_labor_C_95_09, (tot_labor_C_95_09 + tot_capital_C_95_09), out=np.zeros_like(tot_labor_C_95_09), where=(tot_labor_C_95_09 + tot_capital_C_95_09)!=0) # 1107 x 15
    tot_labor_C_95_09 = labor_weight_C_95_09 * VA_95_09.T # 1107 x 15
    tot_capital_C_95_09 = (1-labor_weight_C_95_09) * VA_95_09.T # 1107 x 15
    ### Split labor to different types of labor
    HS_labor_C_95_09 = tot_labor_C_95_09 * np.divide(HS_labor_C_95_09, (HS_labor_C_95_09 + MS_labor_C_95_09 + LS_labor_C_95_09), out=np.zeros_like(HS_labor_C_95_09), where=(HS_labor_C_95_09 + MS_labor_C_95_09 + LS_labor_C_95_09)!=0) # 1107 x 15
    MS_labor_C_95_09 = tot_labor_C_95_09 * np.divide(MS_labor_C_95_09, (HS_labor_C_95_09 + MS_labor_C_95_09 + LS_labor_C_95_09), out=np.zeros_like(MS_labor_C_95_09), where=(HS_labor_C_95_09 + MS_labor_C_95_09 + LS_labor_C_95_09)!=0) # 1107 x 15
    LS_labor_C_95_09 = tot_labor_C_95_09 * np.divide(LS_labor_C_95_09, (HS_labor_C_95_09 + MS_labor_C_95_09 + LS_labor_C_95_09), out=np.zeros_like(LS_labor_C_95_09), where=(HS_labor_C_95_09 + MS_labor_C_95_09 + LS_labor_C_95_09)!=0) # 1107 x 15

    ### Impute ROW average_gos_GFCF_ratio
    ROW_GOS_GFCF = np.sum(temp_weight[:, None] * average_gos_GFCF_ratio, axis=0) # shape: 27 x 1
    average_gos_GFCF_ratio = np.append(average_gos_GFCF_ratio.flatten(), ROW_GOS_GFCF) # shape: 1107 x 1
    ##########################################################
    ### Compute gross operating surplus (= VA - labor compensation)
    gos_95_09 = VA_95_09.T - tot_labor_C_95_09 # shape: 1107 x 5
    ### NOTE: some of these values are negative (should be force them to 0 and re-compute VA?)
    # gos_Ding_data = np.array(CapitalData_sorted.loc[(CapitalData_sorted['finalcode_in'].isin(finalcode)) & (CapitalData_sorted['finalcode_out'].isin(finalcode)), 'rK_njik'])
    # gos_Ding_njik = np.reshape(gos_Ding_data, (Nr, Ns, Nr, Ns)) 
    # gos_Ding = np.sum(gos_Ding_njik, axis=(2,3))
    # gos_use = gos_Ding.copy() 
    gos_use_95_09 = gos_95_09.copy()


    ### Read in depreciation rate and capital stock in the US
    try:
        net_stock_data = pd.read_excel(path+'data_raw/BEA_net_stock.xlsx').iloc[6:-7,2:].to_numpy()
        depreciation_data = pd.read_excel(path+'data_raw/BEA_depreciation.xlsx').iloc[6:-7,2:].to_numpy()
    except FileNotFoundError:
        print(f"File not found")
    except IOError:
        print(f"Error reading file")

    net_stock_data_1997 = 1000 * net_stock_data[:,1997-1947]
    depreciation_data_1997 = 1000 * depreciation_data[:,1997-1947]
    ### Consolidate into 27 sectors
    indices_keep = np.array([1, 4, 29, 32, 33, 15, 34, 35, 36, 37, 38, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 8, 12, 39, 42, 47, 92, 56, 61, 73, 74, 83, 84, 75, 79, 80, 89, 95])

    net_stock_data_27 = net_stock_data_1997[indices_keep]
    depreciation_data_27 = depreciation_data_1997[indices_keep]

    ### paper and printing 
    net_stock_data_27[6] = np.sum(net_stock_data_27[6:8])
    ### primary metal and fabricated metal
    net_stock_data_27[12] = np.sum(net_stock_data_27[12:14])
    ### computer electronics and electrical equipments
    net_stock_data_27[15] = np.sum(net_stock_data_27[15:17])
    ### motor and other transports
    net_stock_data_27[17] = np.sum(net_stock_data_27[17:19])
    ### other manufacturing goods
    net_stock_data_27[19] = np.sum(net_stock_data_27[19:21])
    ### wholesale, retail and transport
    net_stock_data_27[23] = np.sum(net_stock_data_27[23:26])
    ### other professional activities
    net_stock_data_27[33] = np.sum(net_stock_data_27[33:])


    depreciation_data_27[6]  = np.sum(depreciation_data_27[6:8])
    depreciation_data_27[12] = np.sum(depreciation_data_27[12:14])
    depreciation_data_27[15] = np.sum(depreciation_data_27[15:17])
    depreciation_data_27[17] = np.sum(depreciation_data_27[17:19])
    depreciation_data_27[19] = np.sum(depreciation_data_27[19:21])
    depreciation_data_27[23] = np.sum(depreciation_data_27[23:26])
    depreciation_data_27[33] = np.sum(depreciation_data_27[33:])

    net_stock_data_consolidated = np.delete(net_stock_data_27, np.array([7, 13, 16, 18, 20, 24, 25, 34, 35, 36, 37])) # shape: 27 x 1
    depreciation_data_consolidated = np.delete(depreciation_data_27, np.array([7, 13, 16, 18, 20, 24, 25, 34, 35, 36, 37])) # shape: 27 x 1

    depreciation_rate = depreciation_data_consolidated / (depreciation_data_consolidated + net_stock_data_consolidated)
    ### NOTE: depreciation_rate here is different from δ_i in the closed-economy since we combine sectors together

    depreciation_rate_data = depreciation_data_consolidated / (depreciation_data_consolidated + net_stock_data_consolidated)


    
    # Save intermediate results
    with open(path + 'data_intermediate/calibration_intermediate.pkl', 'wb') as f:
        pickle.dump(IO_95_09, f)
        pickle.dump(consumption_95_09, f)
        pickle.dump(inv_flows_njik, f)
        pickle.dump(inv_flows_njik_ratio, f)
        pickle.dump(GFCF_95_09, f)
        pickle.dump(VA_95_09, f)
        pickle.dump(gos_use_95_09, f)
        pickle.dump(HS_labor_C_95_09, f)
        pickle.dump(MS_labor_C_95_09, f)
        pickle.dump(LS_labor_C_95_09, f)
        pickle.dump(average_gos_GFCF_ratio, f)
        pickle.dump(consumption, f)
        pickle.dump(depreciation_rate, f)
        pickle.dump(depreciation_rate_data, f)


    return 0
