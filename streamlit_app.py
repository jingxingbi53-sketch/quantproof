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