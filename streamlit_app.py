import pandas as pd
import streamlit as st

st.set_page_config(page_title="QuantProof", page_icon="📊")

st.title("QuantProof")
st.write("A research-focused platform for auditing quantitative backtests.")

transaction_cost_bps = st.number_input(
    "Transaction cost (basis points)",
    min_value=0,
    max_value=100,
    value=10,
    step=1,
    help="100 basis points equals 1%.",
)

transaction_cost_percent = transaction_cost_bps / 100

st.write(f"Selected transaction cost: {transaction_cost_percent:.2f}%")

trade_value = st.number_input(
	"Trade value ($)",
	min_value = 0.0,
	value = 10000.0,
	step = 1000.0,
)

estimated_transaction_cost = (
	trade_value * transaction_cost_bps / 10000
)

st.metric(
	"Estimated transaction cost",
	f"${estimated_transaction_cost:,.2f}",
	border=True,
)


st.divider()
st.subheader("Upload backtest data")

uploaded_file = st.file_uploader(
	"Choose a CSV file",
	type="csv",
	help="Upload a CSV containing backtest returns.",
)

if uploaded_file is not None:
	st.success(f"Uploaded: {uploaded_file.name}")
	backtest_data = pd.read_csv(uploaded_file)

	required_columns = {"date", "strategy_return"}
	missing_columns = required_columns - set(backtest_data.columns)

	if missing_columns:
		missing_text = ", ".join(sorted(missing_columns))
		st.error(f"Missing required columns: {missing_text}")
	else:
		backtest_data["date"] = pd.to_datetime(
			backtest_data["date"],
			errors="coerce",
		)
		backtest_data["strategy_return"] = pd.to_numeric(
			backtest_data["strategy_return"],
			errors="coerce",
		)
		invalid_rows = backtest_data[
			["date", "strategy_return"]
		].isna().any(axis=1)
		if invalid_rows.any():
			invalid_count = int(invalid_rows.sum())
			st.error(
				f"Found {invalid_count} row(s) with invalid or missing values."
			)
			st.dataframe(
				backtest_data.loc[invalid_rows],
				hide_index=True,
			)
		else:
			st.success("Data validation passed.")
			st.dataframe(backtest_data, hide_index=True)



