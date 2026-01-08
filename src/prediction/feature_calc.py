# line_profile_talib.py
import pandas as pd
import numpy as np
import talib
from talib import abstract


def add_all_talib_features(df):
    df_out = df.copy()
    df_out.columns = [c.lower() for c in df_out.columns]

    new_cols = []

    for func_name in talib.get_functions():
        if func_name == "MAVP":
            continue
        try:
            indicator_fn = abstract.Function(func_name)
            result = indicator_fn(df_out)

            if isinstance(result, pd.Series):
                s = result.rename(func_name)
                new_cols.append(s)
            elif isinstance(result, pd.DataFrame):
                r = result.copy()
                r.columns = [f"{func_name}_{col}" for col in r.columns]
                new_cols.append(r)

        except Exception:
            pass

    if new_cols:
        df_out = pd.concat([df_out] + new_cols, axis=1)

    return df_out


