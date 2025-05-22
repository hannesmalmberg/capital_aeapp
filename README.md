# capital_aeapp

Replication package for "Long-Run Consequences of Sanctions on Russia" (Baqaee and Malmberg, 2025).

## Data Availability Statement

This project uses the following data sources:

1. **Capital Services in Global Value Chains** (Xiang Ding, 2022). These data are not publicly available. The repository contains an empty directory `@Import/data_raw/Ding_data/` as a placeholder. Researchers who wish to replicate the results should contact Xiang Ding.
2. **BEA Net Stock and Depreciation spreadsheets.** Publicly available from the Bureau of Economic Analysis. Save `BEA_net_stock.xlsx` and `BEA_depreciation.xlsx` in `@Import/data_raw/`.
3. **External Wealth of Nations dataset** (Lane and Milesi-Ferretti, 2018). Download the latest version from [Brookings](https://www.brookings.edu/articles/the-external-wealth-of-nations-database/) and store `EWN-dataset.xlsx` in `@Import/data_raw/`.
4. **World Input Output Database** (Timmer et al., 2015). Obtain the Excel tables from the WIOD Dataverse (doi: 10.34894/PJ2M1C) and place them in `@Import/data_raw/WIOD/`.

The authors had legitimate access to all data for this research. Only the Ding data cannot be redistributed. All other data are publicly accessible as described above. Running the provided code will reproduce all derived datasets.

## Repository Contents

- `aea_pp.ipynb` – master notebook that orchestrates the entire replication.
- `clean_data.py` – converts raw data into intermediate pickle files.
- `WIOT_data_cleaning.py` – creates `.mat` files in `@Import/data_intermediate` from the WIOD Excel tables.
- `calibration.py` – calibrates parameters.
- `make_shocks.py`, `solution.py`, and `nonlinear.py` – solve the model and generate results.
- `utils.py` – helper functions.
- `LICENSE` – BSD 3-Clause license.

## Software Requirements

The code requires Python 3.11 and the following packages:

```
pip install numpy pandas scipy numba einops ipython jupyter matplotlib
```

## Instructions to Reproduce Results

1. Create the expected directory structure:

```
mkdir -p '@Import/data_raw/Ding_data' '@Import/data_raw/WIOD' '@Import/data_intermediate'
```

Place the datasets in `@Import/data_raw/` as detailed above. The `Ding_data` folder remains empty unless you obtain those files privately. Save the WIOD Excel tables in the `WIOD` subfolder and the External Wealth of Nations and BEA spreadsheets directly in `data_raw`.

2. Launch Jupyter and run `aea_pp.ipynb` from start to finish. The notebook calls all scripts needed to generate the tables and figures in the paper. Typical running time is about 30 minutes on a standard desktop.

## Mapping Programs to Outputs

- `clean_data.py` prepares the analysis data.
- `calibration.py` and `solution.py` calibrate and solve the model.
- `make_shocks.py` and `nonlinear.py` generate counterfactual simulations.
- All outputs used in the paper are produced when `aea_pp.ipynb` completes.

The package omits the Ding data because it is not public. No other parts have been omitted.
