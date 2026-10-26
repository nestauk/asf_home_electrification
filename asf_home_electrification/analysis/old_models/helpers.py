import pandas as pd


def apply_dst_adjustments(
    df: pd.DataFrame,
    date_col: str,
    timestamp_col: str,
    period_id_col: str,
    clocks_forward_date: pd.Timestamp,
    clocks_back_date: pd.Timestamp,
) -> pd.DataFrame:
    df = df.copy()

    # Clocks forward — remove 23:00 and 23:30 to match Elexon 46-period day
    mask_1 = (df[date_col] == clocks_forward_date) & (df[timestamp_col] == "23:00")
    mask_2 = (df[date_col] == clocks_forward_date) & (df[timestamp_col] == "23:30")
    df = df[~mask_1 & ~mask_2]

    # Clocks forward — fill gaps caused by missing UTC hours (01:00, 01:30 skipped in BST)
    fwd_day = df[df[date_col] == clocks_forward_date]
    expected = set(range(1, 47))
    actual = set(fwd_day[period_id_col].tolist())
    missing = sorted(expected - actual)
    if missing:
        fill_row = fwd_day[fwd_day[period_id_col] == min(missing) - 1].iloc[0]
        extra_rows_fwd = pd.DataFrame([{**fill_row, period_id_col: p} for p in missing])
        df = pd.concat([df, extra_rows_fwd], ignore_index=True)

    # Clocks back — add periods 49 and 50 as duplicate of 00:00 and 00:30
    # to match Elexon 50-period day where the extra hour repeats at start of GMT
    row_0000 = df[(df[date_col] == clocks_back_date) & (df[timestamp_col] == "00:00")].iloc[0]
    row_0030 = df[(df[date_col] == clocks_back_date) & (df[timestamp_col] == "00:30")].iloc[0]

    extra_rows_back = pd.DataFrame(
        [
            {**row_0000, period_id_col: 49, timestamp_col: "00:00"},
            {**row_0030, period_id_col: 50, timestamp_col: "00:30"},
        ]
    )

    df = pd.concat([df, extra_rows_back], ignore_index=True)
    df = df.sort_values([date_col, period_id_col]).reset_index(drop=True)

    return df
