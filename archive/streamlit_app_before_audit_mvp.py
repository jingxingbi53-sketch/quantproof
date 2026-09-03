"""Archived beginner version before the audit-dashboard refactor."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="QuantProof", page_icon="📊")
st.title("QuantProof")
st.write("A research-focused platform for auditing quantitative backtests.")

transaction_cost_bps = st.number_input(
    "Transaction cost (basis points)", min_value=0, max_value=100, value=10, step=1
)
trade_value = st.number_input(
    "Trade value ($)", min_value=0.0, value=10000.0, step=1000.0
)
st.metric(
    "Estimated transaction cost",
    f"${trade_value * transaction_cost_bps / 10000:,.2f}",
    border=True,
)

st.divider()
st.subheader("Upload backtest data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")
    backtest_data = pd.read_csv(uploaded_file)
    required_columns = {"date", "strategy_return"}
    missing_columns = required_columns - set(backtest_data.columns)

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    else:
        backtest_data["date"] = pd.to_datetime(backtest_data["date"], errors="coerce")
        backtest_data["strategy_return"] = pd.to_numeric(
            backtest_data["strategy_return"], errors="coerce"
        )
        invalid_rows = backtest_data[["date", "strategy_return"]].isna().any(axis=1)
        if invalid_rows.any():
            st.error(
                f"Found {int(invalid_rows.sum())} row(s) with invalid or missing values."
            )
            st.dataframe(backtest_data.loc[invalid_rows], hide_index=True)
        else:
            duplicate_dates = backtest_data["date"].duplicated(keep=False)
            if duplicate_dates.any():
                st.error(
                    f"Found {int(duplicate_dates.sum())} row(s) with duplicate dates."
                )
                st.dataframe(backtest_data.loc[duplicate_dates], hide_index=True)
            else:
                st.success("Data validation passed.")
                st.dataframe(backtest_data, hide_index=True)
