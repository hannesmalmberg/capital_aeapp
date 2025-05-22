'''
Data Cleaning on WIOT data in 1997 (everything in millions)

Author: Yutong Zhong

'''

'''
Data Cleaning on WIOT data from 1995-2009 (everything in millions)

Author: Yutong Zhong

'''

#%%
import scipy.io
import numpy as np
import pandas as pd


#%%
### Read in excel data
years = np.arange(1995, 2010)
path_in = '@Import/data_raw/' # path to the data folder
path_out = "@Import/data_intermediate/" 

WIOT1995 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT95_ROW_Apr12.xlsx')
WIOT1996 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT96_ROW_Apr12.xlsx')
WIOT1997 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT97_ROW_Apr12.xlsx')
WIOT1998 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT98_ROW_Apr12.xlsx')
WIOT1999 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT99_ROW_Apr12.xlsx')
WIOT2000 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT00_ROW_Apr12.xlsx')
WIOT2001 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT01_ROW_Apr12.xlsx')
WIOT2002 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT02_ROW_Apr12.xlsx')
WIOT2003 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT03_ROW_Apr12.xlsx')
WIOT2004 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT04_ROW_Apr12.xlsx')
WIOT2005 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT05_ROW_Apr12.xlsx')
WIOT2006 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT06_ROW_Apr12.xlsx')
WIOT2007 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT07_ROW_Apr12.xlsx')
WIOT2008 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT08_ROW_Sep12.xlsx')
WIOT2009 = pd.read_excel(path_in+'WIOT/WIOTS_in_EXCEL/WIOT09_ROW_Sep12.xlsx')


# %%
WIOT_list = [WIOT1995, WIOT1996, WIOT1997, WIOT1998, WIOT1999, WIOT2000, WIOT2001, WIOT2002, WIOT2003, WIOT2004, WIOT2005, WIOT2006, WIOT2007, WIOT2008, WIOT2009]

# %%
Description = pd.read_stata(path_in + 'Ding_data/WIOT_MK_97_finalcode_desc.dta')

#%%
### Extract information
WIOTcode = np.asarray(WIOT1997.iloc[1,4:1439].unique()) # 35 unique sectors
WIOTcountry = np.asarray(WIOT1997.iloc[3,4:1439].unique()) # 41 unique countries
finalcode = np.asarray(Description['finalcode'].unique()) # 27 unique sectors

#%%

Ns, Nf, Nr = 35, 5, 41
## row index
household = 0
non_profit = 1
govt = 2
gfcf = 3
inventory = 4

IO = []
VA = []
grossout = []
final_consumption = []

for i in range(len(years)):
    IO.append(np.asarray(WIOT_list[i].iloc[5:1440,4:1439], dtype = float)) # world input-output table
    VA.append(np.asarray(WIOT_list[i].iloc[1445,4:1439], dtype = float))
    grossout.append(np.asarray(WIOT_list[i].iloc[1447,4:1439], dtype = float))
    final_consumption.append(np.asarray(WIOT_list[i].iloc[5:1440,1439:1644], dtype = float))

#%%
IO = np.array(IO) ## shape: 15 x 1435 x 1435 (first dimension is time)
VA = np.array(VA) ## shape: 15 x 1435 
grossout = np.array(grossout)  ## shape: 15 x 1435 
final_consumption = np.array(final_consumption)  ## shape: 15 x 1435 x 205

#%%
consumption = final_consumption[:,:, household: (Nr-1)*Nf+household+1: Nf] + final_consumption[:,:, non_profit: (Nr-1)*Nf+non_profit+1: Nf] + final_consumption[:,:, govt: (Nr-1)*Nf+govt+1: Nf] ## shape: 15 x 1435 x 41

GFCF = final_consumption[:,:, gfcf: (Nr-1)*Nf+gfcf+1: Nf] ## shape: 15 x 1435 x 41
Inventory = final_consumption[:,:, inventory: (Nr-1)*Nf+inventory+1: Nf] ## shape: 15 x 1435 x 41

#%%
### Consolidate 35 sectors into 27 sectors
'''
O (other community, social and personal services), L (public admin and defense), P (private households with employed persons) --> LOP (other services)

60, 63, 51, 61, 62, 50, 52 --> WRT (wholesale, retail, transportation, repair)
'''
# row (column) index
other_community = 33
public_admin = 30
private_household = 34

repair = 18
trade = 19
motor = 20
inland_transport = 22
water_transport = 23
air_transport = 24
travel = 25

#%%
### Row collapsing
IO[:,other_community: (Nr-1)*Ns+other_community+1: Ns, :] = \
    IO[:,other_community: (Nr-1)*Ns+other_community+1: Ns, :] \
        + IO[:,public_admin: (Nr-1)*Ns+public_admin+1: Ns, :] \
        + IO[:,private_household: (Nr-1)*Ns+private_household+1: Ns, :]
 
IO[:,repair: (Nr-1)*Ns+repair+1: Ns, :] = \
    IO[:,repair: (Nr-1)*Ns+repair+1: Ns, :] \
        + IO[:,trade: (Nr-1)*Ns+trade+1: Ns, :] \
        + IO[:,motor: (Nr-1)*Ns+motor+1: Ns, :] \
        + IO[:,inland_transport: (Nr-1)*Ns+inland_transport+1: Ns, :] \
        + IO[:,water_transport: (Nr-1)*Ns+water_transport+1: Ns, :] \
        + IO[:,air_transport: (Nr-1)*Ns+air_transport+1: Ns, :] \
        + IO[:,travel: (Nr-1)*Ns+travel+1: Ns, :] 

### Column collapsing
IO[:,:, other_community: (Nr-1)*Ns+other_community+1: Ns] = \
    IO[:,:, other_community: (Nr-1)*Ns+other_community+1: Ns] \
        + IO[:,:, public_admin: (Nr-1)*Ns+public_admin+1: Ns] \
        + IO[:,:, private_household: (Nr-1)*Ns+private_household+1: Ns]

IO[:,:, repair: (Nr-1)*Ns+repair+1: Ns] = \
    IO[:,:, repair: (Nr-1)*Ns+repair+1: Ns] \
        + IO[:,:, trade: (Nr-1)*Ns+trade+1: Ns] \
        + IO[:,:, motor: (Nr-1)*Ns+motor+1: Ns] \
        + IO[:,:, inland_transport: (Nr-1)*Ns+inland_transport+1: Ns] \
        + IO[:,:, water_transport: (Nr-1)*Ns+water_transport+1: Ns] \
        + IO[:,:, air_transport: (Nr-1)*Ns+air_transport+1: Ns] \
        + IO[:,:, travel: (Nr-1)*Ns+travel+1: Ns] 
#%%
### delete irrelevant rows and columns (new shape: 15 x 1107 x 1107)
IO = np.delete(IO, np.array(
    [
        np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns), 
        np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns), 
        np.arange(trade, Ns*(Nr-1)+trade+1,Ns), 
        np.arange(motor, Ns*(Nr-1)+motor+1,Ns), 
        np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns), 
        np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns), 
        np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns), 
        np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
        ]).flatten(), axis = 1)

IO = np.delete(IO, np.array(
    [
        np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns), 
        np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns), 
        np.arange(trade, Ns*(Nr-1)+trade+1,Ns), 
        np.arange(motor, Ns*(Nr-1)+motor+1,Ns), 
        np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns), 
        np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns), 
        np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns), 
        np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
        ]).flatten(), axis = 2)

# %%
### collapsing value added, grossout from 15 x 1435 to 15 x 1107 vectors
### collapsing Inventory, consumption, GFCF from 15 x 1435 x 41 to 15 x 1107 x 41 matrices

vec_list = [VA.copy(), grossout.copy()]
mat_list = [Inventory.copy(), consumption.copy(), GFCF.copy()]

for i, vec in enumerate(vec_list):
    vec[:,other_community: (Nr-1)*Ns+other_community+1: Ns] = \
        vec[:,other_community: (Nr-1)*Ns+other_community+1: Ns] \
            + vec[:,public_admin: (Nr-1)*Ns+public_admin+1: Ns] \
            + vec[:,private_household: (Nr-1)*Ns+private_household+1: Ns]

    vec[:,repair: (Nr-1)*Ns+repair+1: Ns] = \
        vec[:,repair: (Nr-1)*Ns+repair+1: Ns] \
            + vec[:,trade: (Nr-1)*Ns+trade+1: Ns] \
            + vec[:,motor: (Nr-1)*Ns+motor+1: Ns] \
            + vec[:,inland_transport: (Nr-1)*Ns+inland_transport+1: Ns] \
            + vec[:,water_transport: (Nr-1)*Ns+water_transport+1: Ns] \
            + vec[:,air_transport: (Nr-1)*Ns+air_transport+1: Ns] \
            + vec[:,travel: (Nr-1)*Ns+travel+1: Ns]

    vec_list[i] = np.delete(vec,
        np.array(
            [
                np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns),
                np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns),
                np.arange(trade, Ns*(Nr-1)+trade+1,Ns),
                np.arange(motor, Ns*(Nr-1)+motor+1,Ns),
                np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns),
                np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns),
                np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns),
                np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
            ]
        ).flatten(), axis=1
    )

for i, mat in enumerate(mat_list):
    mat[:,other_community: (Nr-1)*Ns+other_community+1: Ns,:] = \
        mat[:,other_community: (Nr-1)*Ns+other_community+1: Ns,:] \
            + mat[:,public_admin: (Nr-1)*Ns+public_admin+1: Ns,:] \
            + mat[:,private_household: (Nr-1)*Ns+private_household+1: Ns,:]

    mat[:,repair: (Nr-1)*Ns+repair+1: Ns,:] = \
        mat[:,repair: (Nr-1)*Ns+repair+1: Ns,:] \
            + mat[:,trade: (Nr-1)*Ns+trade+1: Ns,:] \
            + mat[:,motor: (Nr-1)*Ns+motor+1: Ns,:] \
            + mat[:,inland_transport: (Nr-1)*Ns+inland_transport+1: Ns,:] \
            + mat[:,water_transport: (Nr-1)*Ns+water_transport+1: Ns,:] \
            + mat[:,air_transport: (Nr-1)*Ns+air_transport+1: Ns,:] \
            + mat[:,travel: (Nr-1)*Ns+travel+1: Ns,:]

    mat_list[i] = np.delete(mat,
        np.array(
            [
                np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns),
                np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns),
                np.arange(trade, Ns*(Nr-1)+trade+1,Ns),
                np.arange(motor, Ns*(Nr-1)+motor+1,Ns),
                np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns),
                np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns),
                np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns),
                np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
            ]
        ).flatten(), axis=1
    )
# %%
### Export values to mat file
mdic = {'VA': vec_list[0], 'grossout': vec_list[1], 'Inventory': mat_list[0], 'IO': IO, 'consumption': mat_list[1], 'GFCF': mat_list[2]}
scipy.io.savemat(path_out+"WIOT95_09.mat", mdic)



#%%
### Read in excel data for WIOT SEA data (everything in millions)
years = np.arange(1995, 2010)
try:
    SEA = pd.read_excel(path_in+'WIOT/Socio_Economic_Accounts_July14.xlsx', sheet_name='DATA')
    SEA_data = SEA.loc[:, ['Country', 'Variable', 'Code', *[f'_{year}' for year in years]]]

except FileNotFoundError:
    print(f"File not found: {'Socio_Economic_Accounts_July14.xlsx'}")
except IOError:
    print(f"Error reading file: {'Socio_Economic_Accounts_July14.xlsx'}")


# %%
### Extract labor compensation data
# variables = np.array(['LAB', 'LABHS', 'LABMS', 'LABLS', 'GO', 'CAP'])
variables = np.array(['COMP', 'LABHS', 'LABMS', 'LABLS', 'GO', 'CAP', 'VA'])
WIOTcode = np.array(['AtB', 'C', '15t16', '17t18', '19', '20', '21t22', '23', '24', '25', '26',
       '27t28', '29', '30t33', '34t35', '36t37', 'E', 'F', '50', '51', '52', 'H',
       '60', '61', '62', '63', '64', 'J', '70', '71t74', 'L', 'M', 'N', 'O', 'P'])

SEA_relevant = SEA_data.loc[SEA_data['Variable'].isin(variables) & SEA_data['Code'].isin(WIOTcode)] # 11200 rows (40 countries (no ROW) x 35 sectors x 8 variables)


# %%
Ns, Nr, Nf = 35, 40, 5
labor_compensation = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'COMP', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 5
labor_compensation[np.isnan(labor_compensation)] = 0

HS_share = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'LABHS', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 5
HS_share[np.isnan(HS_share)] = 0

MS_share = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'LABMS', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 5
MS_share[np.isnan(MS_share)] = 0

LS_share = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'LABLS', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 5
LS_share[np.isnan(LS_share)] = 0

# %%
HS_compensation = labor_compensation * HS_share
MS_compensation = labor_compensation * MS_share
LS_compensation = labor_compensation * LS_share

# %%
### gross output data by sector in national currency (millions)
gross_output_national = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'GO', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 5
gross_output_national[np.isnan(gross_output_national)] = 0

### capital compensation by sector in national currency
capital_comp = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'CAP', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 5
# %%
### Collapse into 27 sectors (vector of len 1400 --> vector of len 1080)
# row (column) index
other_community = 33
public_admin = 30
private_household = 34

repair = 18
trade = 19
motor = 20
inland_transport = 22
water_transport = 23
air_transport = 24
travel = 25

labor_list = [HS_compensation.copy(), MS_compensation.copy(), LS_compensation.copy(), labor_compensation.copy(), gross_output_national.copy(), capital_comp.copy()]

# %%
for i, vec in enumerate(labor_list):
    vec[other_community: (Nr-1)*Ns+other_community+1: Ns,:] = \
        vec[other_community: (Nr-1)*Ns+other_community+1: Ns,:] \
            + vec[public_admin: (Nr-1)*Ns+public_admin+1: Ns,:] \
            + vec[private_household: (Nr-1)*Ns+private_household+1: Ns,:]

    vec[repair: (Nr-1)*Ns+repair+1: Ns,:] = \
        vec[repair: (Nr-1)*Ns+repair+1: Ns,:] \
            + vec[trade: (Nr-1)*Ns+trade+1: Ns,:] \
            + vec[motor: (Nr-1)*Ns+motor+1: Ns,:] \
            + vec[inland_transport: (Nr-1)*Ns+inland_transport+1: Ns,:] \
            + vec[water_transport: (Nr-1)*Ns+water_transport+1: Ns,:] \
            + vec[air_transport: (Nr-1)*Ns+air_transport+1: Ns,:] \
            + vec[travel: (Nr-1)*Ns+travel+1: Ns,:]

    labor_list[i] = np.delete(vec,
        np.array(
            [
                np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns),
                np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns),
                np.arange(trade, Ns*(Nr-1)+trade+1,Ns),
                np.arange(motor, Ns*(Nr-1)+motor+1,Ns),
                np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns),
                np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns),
                np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns),
                np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
            ]
        ).flatten(), axis=0
    )
### shape: 1080 x 15 after deletion
# %%
### Export values to mat file
mdic1 = {'HS_labor_95_09': labor_list[0], 'MS_labor_95_09': labor_list[1], 'LS_labor_95_09': labor_list[2], 'tot_labor_95_09': labor_list[3], 'grossout_national_95_09': labor_list[4], 'capital_comp_95_09': labor_list[5]}
scipy.io.savemat(path_out+"SEA95_09.mat", mdic1)
# %%
grossout_national = np.reshape(labor_list[4].T, (15, 40, 27))
grossout_exclude_ROW = np.reshape(vec_list[1][:,:1080], (15, 40, 27))
currency_exchange = np.divide(grossout_national, grossout_exclude_ROW, out=np.zeros_like(grossout_national), where=grossout_exclude_ROW!=0)
### should be the same across sectors for a country
# %%



#%%
### Do special case for 1997

try:
    WIOT1997 = pd.read_excel(path_in + 'WIOT/WIOTS_in_EXCEL/WIOT97_ROW_Apr12.xlsx')
    Description = pd.read_stata(path_in + 'Ding_data/WIOT_MK_97_finalcode_desc.dta')

except FileNotFoundError:
    print(f"File not found: {'WIOT97_ROW_Apr12.xlsx'}")
except IOError:
    print(f"Error reading file: {'WIOT97_ROW_Apr12.xlsx'}")

#%%
### Extract information
WIOTcode = np.asarray(WIOT1997.iloc[1,4:1439].unique()) # 35 unique sectors
WIOTcountry = np.asarray(WIOT1997.iloc[3,4:1439].unique()) # 41 unique countries
finalcode = np.asarray(Description['finalcode'].unique()) # 27 unique sectors
IO = np.asarray(WIOT1997.iloc[5:1440,4:1439], dtype = float) # world input-output table
VA = np.asarray(WIOT1997.iloc[1445,4:1439], dtype = float)
grossout = np.asarray(WIOT1997.iloc[1447,4:1439], dtype = float)

#%%
### Extract final consumption
Ns, Nf, Nr = 35, 5, 41
## row index
household = 0
non_profit = 1
govt = 2
gfcf = 3
inventory = 4
final_consumption = np.asarray(WIOT1997.iloc[5:1440,1439:1644], dtype = float)

consumption = final_consumption[:, household: (Nr-1)*Nf+household+1: Nf] + final_consumption[:, non_profit: (Nr-1)*Nf+non_profit+1: Nf] + final_consumption[:, govt: (Nr-1)*Nf+govt+1: Nf]

GFCF = final_consumption[:, gfcf: (Nr-1)*Nf+gfcf+1: Nf]
Inventory = final_consumption[:, inventory: (Nr-1)*Nf+inventory+1: Nf]

#%%
### Consolidate 35 sectors into 27 sectors
'''
O (other community, social and personal services), L (public admin and defense), P (private households with employed persons) --> LOP (other services)

60, 63, 51, 61, 62, 50, 52 --> WRT (wholesale, retail, transportation, repair)
'''
# row (column) index
other_community = 33
public_admin = 30
private_household = 34

repair = 18
trade = 19
motor = 20
inland_transport = 22
water_transport = 23
air_transport = 24
travel = 25

#%%
### Row collapsing
IO[other_community: (Nr-1)*Ns+other_community+1: Ns, :] = \
    IO[other_community: (Nr-1)*Ns+other_community+1: Ns, :] \
        + IO[public_admin: (Nr-1)*Ns+public_admin+1: Ns, :] \
        + IO[private_household: (Nr-1)*Ns+private_household+1: Ns, :]
 
IO[repair: (Nr-1)*Ns+repair+1: Ns, :] = \
    IO[repair: (Nr-1)*Ns+repair+1: Ns, :] \
        + IO[trade: (Nr-1)*Ns+trade+1: Ns, :] \
        + IO[motor: (Nr-1)*Ns+motor+1: Ns, :] \
        + IO[inland_transport: (Nr-1)*Ns+inland_transport+1: Ns, :] \
        + IO[water_transport: (Nr-1)*Ns+water_transport+1: Ns, :] \
        + IO[air_transport: (Nr-1)*Ns+air_transport+1: Ns, :] \
        + IO[travel: (Nr-1)*Ns+travel+1: Ns, :] 

### Column collapsing
IO[:, other_community: (Nr-1)*Ns+other_community+1: Ns] = \
    IO[:, other_community: (Nr-1)*Ns+other_community+1: Ns] \
        + IO[:, public_admin: (Nr-1)*Ns+public_admin+1: Ns] \
        + IO[:, private_household: (Nr-1)*Ns+private_household+1: Ns]

IO[:, repair: (Nr-1)*Ns+repair+1: Ns] = \
    IO[:, repair: (Nr-1)*Ns+repair+1: Ns] \
        + IO[:, trade: (Nr-1)*Ns+trade+1: Ns] \
        + IO[:, motor: (Nr-1)*Ns+motor+1: Ns] \
        + IO[:, inland_transport: (Nr-1)*Ns+inland_transport+1: Ns] \
        + IO[:, water_transport: (Nr-1)*Ns+water_transport+1: Ns] \
        + IO[:, air_transport: (Nr-1)*Ns+air_transport+1: Ns] \
        + IO[:, travel: (Nr-1)*Ns+travel+1: Ns] 
#%%
### delete irrelevant rows and columns (new shape: 1107 x 1107)
IO = np.delete(IO, np.array(
    [
        np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns), 
        np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns), 
        np.arange(trade, Ns*(Nr-1)+trade+1,Ns), 
        np.arange(motor, Ns*(Nr-1)+motor+1,Ns), 
        np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns), 
        np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns), 
        np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns), 
        np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
        ]).flatten(), axis = 0)

IO = np.delete(IO, np.array(
    [
        np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns), 
        np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns), 
        np.arange(trade, Ns*(Nr-1)+trade+1,Ns), 
        np.arange(motor, Ns*(Nr-1)+motor+1,Ns), 
        np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns), 
        np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns), 
        np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns), 
        np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
        ]).flatten(), axis = 1)

# %%
### collapsing value added, grossout from 1435 x 1 to 1107 x 1 vectors
### collapsing Inventory, consumption, GFCF from 1435 x 41 to 1107 x 41 matrices

vec_list = [VA.copy(), grossout.copy()]
mat_list = [Inventory.copy(), consumption.copy(), GFCF.copy()]

for i, vec in enumerate(vec_list):
    vec[other_community: (Nr-1)*Ns+other_community+1: Ns] = \
        vec[other_community: (Nr-1)*Ns+other_community+1: Ns] \
            + vec[public_admin: (Nr-1)*Ns+public_admin+1: Ns] \
            + vec[private_household: (Nr-1)*Ns+private_household+1: Ns]

    vec[repair: (Nr-1)*Ns+repair+1: Ns] = \
        vec[repair: (Nr-1)*Ns+repair+1: Ns] \
            + vec[trade: (Nr-1)*Ns+trade+1: Ns] \
            + vec[motor: (Nr-1)*Ns+motor+1: Ns] \
            + vec[inland_transport: (Nr-1)*Ns+inland_transport+1: Ns] \
            + vec[water_transport: (Nr-1)*Ns+water_transport+1: Ns] \
            + vec[air_transport: (Nr-1)*Ns+air_transport+1: Ns] \
            + vec[travel: (Nr-1)*Ns+travel+1: Ns]

    vec_list[i] = np.delete(vec,
        np.array(
            [
                np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns),
                np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns),
                np.arange(trade, Ns*(Nr-1)+trade+1,Ns),
                np.arange(motor, Ns*(Nr-1)+motor+1,Ns),
                np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns),
                np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns),
                np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns),
                np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
            ]
        ).flatten()
    )

for i, mat in enumerate(mat_list):
    mat[other_community: (Nr-1)*Ns+other_community+1: Ns,:] = \
        mat[other_community: (Nr-1)*Ns+other_community+1: Ns,:] \
            + mat[public_admin: (Nr-1)*Ns+public_admin+1: Ns,:] \
            + mat[private_household: (Nr-1)*Ns+private_household+1: Ns,:]

    mat[repair: (Nr-1)*Ns+repair+1: Ns,:] = \
        mat[repair: (Nr-1)*Ns+repair+1: Ns,:] \
            + mat[trade: (Nr-1)*Ns+trade+1: Ns,:] \
            + mat[motor: (Nr-1)*Ns+motor+1: Ns,:] \
            + mat[inland_transport: (Nr-1)*Ns+inland_transport+1: Ns,:] \
            + mat[water_transport: (Nr-1)*Ns+water_transport+1: Ns,:] \
            + mat[air_transport: (Nr-1)*Ns+air_transport+1: Ns,:] \
            + mat[travel: (Nr-1)*Ns+travel+1: Ns,:]

    mat_list[i] = np.delete(mat,
        np.array(
            [
                np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns),
                np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns),
                np.arange(trade, Ns*(Nr-1)+trade+1,Ns),
                np.arange(motor, Ns*(Nr-1)+motor+1,Ns),
                np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns),
                np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns),
                np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns),
                np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
            ]
        ).flatten(), axis = 0
    )
# %%
### Export values to mat file
mdic = {'VA': vec_list[0], 'grossout': vec_list[1], 'Inventory': mat_list[0], 'IO': IO, 'consumption': mat_list[1], 'GFCF': mat_list[2]}
scipy.io.savemat(path_out + "WIOT1997.mat", mdic)

#%%
### Read in excel data for WIOT SEA data (everything in millions)
years = np.arange(1995, 2010)
try:
    SEA = pd.read_excel(path_in + 'WIOT/Socio_Economic_Accounts_July14.xlsx', sheet_name='DATA')
    SEA_data = SEA.loc[:, ['Country', 'Variable', 'Code', *[f'_{year}' for year in years]]]

except FileNotFoundError:
    print(f"File not found: {'Socio_Economic_Accounts_July14.xlsx'}")
except IOError:
    print(f"Error reading file: {'Socio_Economic_Accounts_July14.xlsx'}")


# %%
### Extract labor compensation data
# variables = np.array(['LAB', 'LABHS', 'LABMS', 'LABLS', 'GO', 'CAP'])
variables = np.array(['COMP', 'LABHS', 'LABMS', 'LABLS', 'GO', 'CAP', 'VA', 'GFCF'])
WIOTcode = np.array(['AtB', 'C', '15t16', '17t18', '19', '20', '21t22', '23', '24', '25', '26',
       '27t28', '29', '30t33', '34t35', '36t37', 'E', 'F', '50', '51', '52', 'H',
       '60', '61', '62', '63', '64', 'J', '70', '71t74', 'L', 'M', 'N', 'O', 'P'])

SEA_relevant = SEA_data.loc[SEA_data['Variable'].isin(variables) & SEA_data['Code'].isin(WIOTcode)] # 11200 rows (40 countries (no ROW) x 35 sectors x 8 variables)


# %%
Ns, Nr, Nf = 35, 40, 5
# labor_compensation = np.asarray(SEA1997.loc[SEA1997['Variable'] == 'LAB', '_1997'], dtype=float)
labor_compensation = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'COMP', '_1997'], dtype=float) ## shape: 1400 x 1
labor_compensation[np.isnan(labor_compensation)] = 0

HS_share = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'LABHS', '_1997'], dtype=float) ## shape: 1400 x 1
HS_share[np.isnan(HS_share)] = 0

MS_share = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'LABMS', '_1997'], dtype=float) ## shape: 1400 x 1
MS_share[np.isnan(MS_share)] = 0

LS_share = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'LABLS', '_1997'], dtype=float) ## shape: 1400 x 1
LS_share[np.isnan(LS_share)] = 0


# %%
HS_compensation = labor_compensation * HS_share
MS_compensation = labor_compensation * MS_share
LS_compensation = labor_compensation * LS_share

# %%
### gross output data by sector in national currency (millions)
gross_output_national = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'GO', '_1997'], dtype=float)
gross_output_national[np.isnan(gross_output_national)] = 0

### capital compensation by sector in national currency
capital_comp = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'CAP', '_1997'], dtype=float)
# %%
### Collapse into 27 sectors (vector of len 1400 --> vector of len 1080)
# row (column) index
other_community = 33
public_admin = 30
private_household = 34

repair = 18
trade = 19
motor = 20
inland_transport = 22
water_transport = 23
air_transport = 24
travel = 25

labor_list = [HS_compensation.copy(), MS_compensation.copy(), LS_compensation.copy(), labor_compensation.copy(), gross_output_national.copy(), capital_comp.copy()]

for i, vec in enumerate(labor_list):
    vec[other_community: (Nr-1)*Ns+other_community+1: Ns] = \
        vec[other_community: (Nr-1)*Ns+other_community+1: Ns] \
            + vec[public_admin: (Nr-1)*Ns+public_admin+1: Ns] \
            + vec[private_household: (Nr-1)*Ns+private_household+1: Ns]

    vec[repair: (Nr-1)*Ns+repair+1: Ns] = \
        vec[repair: (Nr-1)*Ns+repair+1: Ns] \
            + vec[trade: (Nr-1)*Ns+trade+1: Ns] \
            + vec[motor: (Nr-1)*Ns+motor+1: Ns] \
            + vec[inland_transport: (Nr-1)*Ns+inland_transport+1: Ns] \
            + vec[water_transport: (Nr-1)*Ns+water_transport+1: Ns] \
            + vec[air_transport: (Nr-1)*Ns+air_transport+1: Ns] \
            + vec[travel: (Nr-1)*Ns+travel+1: Ns]

    labor_list[i] = np.delete(vec,
        np.array(
            [
                np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns),
                np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns),
                np.arange(trade, Ns*(Nr-1)+trade+1,Ns),
                np.arange(motor, Ns*(Nr-1)+motor+1,Ns),
                np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns),
                np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns),
                np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns),
                np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
            ]
        ).flatten()
    )

# %%
### compute gos/investment for country-sector pair in each year
value_added_years = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'VA', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 16
value_added_years[np.isnan(value_added_years)] = 0

labor_compensation_years = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'COMP', [*[f'_{year}' for year in years]]], dtype=float)
labor_compensation_years[np.isnan(labor_compensation_years)] = 0

gos_years = value_added_years - labor_compensation_years

GFCF_years = np.asarray(SEA_relevant.loc[SEA_relevant['Variable'] == 'GFCF', [*[f'_{year}' for year in years]]], dtype=float) ## shape: 1400 x 16
GFCF_years[np.isnan(GFCF_years)] = 0

investment_list = [gos_years.copy(), GFCF_years.copy()]

for i, vec in enumerate(investment_list):
    vec[other_community: (Nr-1)*Ns+other_community+1: Ns, :] = \
        vec[other_community: (Nr-1)*Ns+other_community+1: Ns, :] \
            + vec[public_admin: (Nr-1)*Ns+public_admin+1: Ns, :] \
            + vec[private_household: (Nr-1)*Ns+private_household+1: Ns, :]

    vec[repair: (Nr-1)*Ns+repair+1: Ns, :] = \
        vec[repair: (Nr-1)*Ns+repair+1: Ns, :] \
            + vec[trade: (Nr-1)*Ns+trade+1: Ns, :] \
            + vec[motor: (Nr-1)*Ns+motor+1: Ns, :] \
            + vec[inland_transport: (Nr-1)*Ns+inland_transport+1: Ns, :] \
            + vec[water_transport: (Nr-1)*Ns+water_transport+1: Ns, :] \
            + vec[air_transport: (Nr-1)*Ns+air_transport+1: Ns, :] \
            + vec[travel: (Nr-1)*Ns+travel+1: Ns, :]

    investment_list[i] = np.delete(vec,
        np.array(
            [
                np.arange(public_admin,Ns*(Nr-1)+public_admin+1,Ns),
                np.arange(private_household,Ns*(Nr-1)+private_household+1,Ns),
                np.arange(trade, Ns*(Nr-1)+trade+1,Ns),
                np.arange(motor, Ns*(Nr-1)+motor+1,Ns),
                np.arange(inland_transport, Ns*(Nr-1)+inland_transport+1,Ns),
                np.arange(water_transport, Ns*(Nr-1)+water_transport+1,Ns),
                np.arange(air_transport, Ns*(Nr-1)+air_transport+1,Ns),
                np.arange(travel, Ns*(Nr-1)+travel+1,Ns)
            ]
        ).flatten(), axis=0 ## delete the rows, not columns
    )
gos_consolidated_years, GFCF_consolidated_years = investment_list

gos_over_GFCF_ratio_years = np.divide(gos_consolidated_years, GFCF_consolidated_years, out=np.zeros_like(gos_consolidated_years), where=GFCF_consolidated_years!=0) ## shape: 1080 x 16

gos_over_GFCF_ratio_mean = np.mean(gos_over_GFCF_ratio_years, axis=1) ## shape: 1080 x 1

# %%
### Export values to mat file
mdic1 = {'HS_labor': labor_list[0], 'MS_labor': labor_list[1], 'LS_labor': labor_list[2],
         'tot_labor': labor_list[3], 'grossout_national': labor_list[4],
         'capital_comp': labor_list[5], 'average_gos_GFCF_ratio': gos_over_GFCF_ratio_mean}
scipy.io.savemat(path_out + "SEA1997.mat", mdic1)