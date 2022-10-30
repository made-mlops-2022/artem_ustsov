"""Copyright 2022 by Artem Ustsov"""

import pandas as pd
from pandas_profiling import ProfileReport


def make_eda_report(input_data_path: str, output_report_path: str):
    df = pd.read_csv(input_data_path)
    profile = ProfileReport(df, title="EDA Report")
    profile.to_file(output_report_path)
