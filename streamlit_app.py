import pandas as pd
import streamlit as st

from analytics import calculate_metrics, cost_sensitivity, prepare_backtest


st.set_page_config(page_title="QuantProof", page_icon=":material/monitoring:", layout="wide")

st.title("QuantProof")
st.write("Answer one question: **Can this backtest be trusted?**")
st.caption("Upload daily returns to receive a transparent audit, not an investment recommendation.")

with st.container(border=True):
    st.subheader("Assumptions")
    transaction_cost_bps = st.number_input(
        "Transaction cost (basis points)",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        help="Applied to daily turnover when the optional turnover column exists.",
    )
    trade_value = st.number_input(
        "Example trade value ($)", min_value=0.0, value=10000.0, step=1000.0
    )
    st.metric(
        "Estimated one-way transaction cost",
        f"${trade_value * transaction_cost_bps / 10000:,.2f}",
        border=True,
    )

st.subheader("Upload backtest data")
st.caption(
    "Required: date, strategy_return. Optional: turnover. "
    "Use decimals: write 1% as 0.01."
)

template = pd.DataFrame(
    {
        "date": ["2026-01-02", "2026-01-05"],
        "strategy_return": [0.01, -0.005],
        "turnover": [0.40, 0.25],
    }
)
st.download_button(
    "Download CSV template",
    template.to_csv(index=False),
    "quantproof_template.csv",
    "text/csv",
    icon=":material/download:",
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"The CSV could not be read: {error}")
    else:
        audit = prepare_backtest(raw_data, transaction_cost_bps)

        st.subheader("Audit result")
        st.metric(
            "QuantProof audit score",
            f"{audit.score}/100 — {audit.verdict}",
            border=True,
        )
        check_table = pd.DataFrame(
            {
                "Status": [check.status for check in audit.checks],
                "Check": [check.name for check in audit.checks],
                "Finding": [check.detail for check in audit.checks],
                "Points": [
                    f"{check.points}/{check.possible_points}" for check in audit.checks
                ],
            }
        )
        st.dataframe(check_table, hide_index=True)

        if audit.blocking_errors:
            for message in audit.blocking_errors:
                st.error(message)
            if not audit.problem_rows.empty:
                st.dataframe(audit.problem_rows, hide_index=True)
        else:
            backtest_data = audit.data
            metrics = calculate_metrics(backtest_data["net_return"])

            st.subheader("Performance snapshot")
            with st.container(horizontal=True):
                st.metric("Total return", f"{metrics['total_return']:.2%}", border=True)
                st.metric(
                    "Annualized return",
                    f"{metrics['annualized_return']:.2%}",
                    border=True,
                )
                st.metric(
                    "Annualized volatility",
                    f"{metrics['annualized_volatility']:.2%}",
                    border=True,
                )
                st.metric(
                    "Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}", border=True
                )
                st.metric(
                    "Maximum drawdown",
                    f"{metrics['max_drawdown']:.2%}",
                    border=True,
                )
                st.metric("Win rate", f"{metrics['win_rate']:.2%}", border=True)

            st.caption(
                "Annualized figures assume 252 trading days and a 0% risk-free rate."
            )
            indexed_data = backtest_data.set_index("date")
            tabs = st.tabs(
                ["Equity curve", "Drawdown", "Cost stress test", "Validated data"]
            )

            with tabs[0]:
                st.line_chart(
                    indexed_data, y="equity_curve", y_label="Growth of $1"
                )

            with tabs[1]:
                st.area_chart(indexed_data, y="drawdown", y_label="Drawdown")

            with tabs[2]:
                if "turnover" in raw_data.columns:
                    sensitivity = cost_sensitivity(
                        backtest_data, [0, 5, 10, 15, 20, 30, 50]
                    )
                    st.line_chart(
                        sensitivity, x="Cost (bps)", y="Total return"
                    )
                    st.dataframe(
                        sensitivity,
                        column_config={
                            "Total return": st.column_config.NumberColumn(
                                format="percent"
                            )
                        },
                        hide_index=True,
                    )
                else:
                    st.info(
                        "Add a turnover column to run the transaction-cost stress test."
                    )

            with tabs[3]:
                st.dataframe(
                    backtest_data,
                    column_config={
                        "date": st.column_config.DateColumn(
                            "Date", format="YYYY-MM-DD"
                        ),
                        "strategy_return": st.column_config.NumberColumn(
                            "Gross return", format="percent"
                        ),
                        "turnover": st.column_config.NumberColumn(
                            "Turnover", format="percent"
                        ),
                        "net_return": st.column_config.NumberColumn(
                            "Net return", format="percent"
                        ),
                        "equity_curve": st.column_config.NumberColumn(
                            "Growth of $1", format="$%.4f"
                        ),
                        "drawdown": st.column_config.NumberColumn(
                            "Drawdown", format="percent"
                        ),
                    },
                    hide_index=True,
                )
