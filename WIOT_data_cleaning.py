"""Prepare WIOD intermediate files.

This script converts the raw World Input Output Database (WIOD)
Excel tables into the `.mat` files consumed by ``clean_data.py``.
It expects the raw WIOD files to be saved in ``@Import/data_raw``
and writes the converted output to ``@Import/data_intermediate``.

The raw data can be obtained from the WIOD Dataverse
(doi: 10.34894/PJ2M1C) as described in the README.

"""
import os
import pandas as pd
import scipy.io


def main():
    raise NotImplementedError(
        "Fill in data conversion from WIOD Excel files to .mat format")


if __name__ == "__main__":
    main()
