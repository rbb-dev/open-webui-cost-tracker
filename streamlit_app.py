"""
Streamlit App for Cost Tracker (Open WebUI function) Data Visualization

This Streamlit application processes and visualizes cost data from a JSON file.
It generates plots for total tokens used and total costs by model and user.

Author: bgeneto
Version: 0.2.2
Date: 2024-11-29
"""

import datetime
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config as the first Streamlit command
st.set_page_config(page_title="Cost Tracker App", page_icon="💵")


@st.cache_data
def load_data(file: Any) -> Optional[List[Dict[str, Any]]]:
    """Load data from a JSON file.

    Args:
        file: A file-like object containing JSON data.

    Returns:
        A list of dictionaries with cost records if the JSON is valid, otherwise None.
    """
    try:
        data = json.load(file)
        return data
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON file.")
        return None


def process_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process the data by extracting the month, model, cost, and user.

    Args:
        data: A list of dictionaries containing cost records.

    Returns:
        A pandas DataFrame with processed data.
    """
    processed_data = []
    # Iterate through each record in the list
    for record in data:
        try:
            timestamp = datetime.datetime.strptime(
                record["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"
            )
            month = timestamp.strftime("%Y-%m")
            model = record["model"]
            cost = record["total_cost"]
            user_email = record["user"] # Get user email directly from the record

            # Ensure cost is a float
            if isinstance(cost, str):
                cost = float(cost)

            total_tokens = record["input_tokens"] + record["output_tokens"]

            processed_data.append(
                {
                    "month": month,
                    "model": model,
                    "total_cost": cost,
                    "user": user_email,
                    "total_tokens": total_tokens,
                }
            )
        except KeyError as e:
            st.error(f"Missing key in record: {e}. Record: {record}")
            continue
        except ValueError:
            st.error(f"Invalid cost or token value in record: {record}")
            continue
        except Exception as e:
            st.error(f"An error occurred processing record: {record}. Error: {e}")
            continue

    if not processed_data:
        st.warning("No valid data found to process.")
        # Return an empty DataFrame with expected columns to avoid downstream errors
        return pd.DataFrame(columns=["month", "model", "total_cost", "user", "total_tokens"])

    return pd.DataFrame(processed_data)


def plot_data(data: pd.DataFrame, month: str) -> None:
    """Plot the data for a specific month.

    Args:
        data: A pandas DataFrame containing processed data.
        month: A string representing the month to filter data.
    """
    month_data = data[data["month"] == month]

    if month_data.empty:
        st.error(f"No data available for {month}.")
        return

    # ---------------------------------
    # Model Usage Bar Plot (Total Tokens)
    # ---------------------------------
    month_data_models_tokens = (
        month_data.groupby("model")["total_tokens"].sum().reset_index()
    )
    month_data_models_tokens = month_data_models_tokens.sort_values(
        by="total_tokens", ascending=False
    ).head(10)
    fig_models_tokens = px.bar(
        month_data_models_tokens,
        x="model",
        y="total_tokens",
        title=f"Top 10 Total Tokens Used by Model ({month})",
    )
    st.plotly_chart(fig_models_tokens, use_container_width=True)

    # ---------------------------------
    # Model Cost Bar Plot (Total Cost)
    # ---------------------------------
    month_data_models_cost = (
        month_data.groupby("model")["total_cost"].sum().reset_index()
    )
    month_data_models_cost = month_data_models_cost.sort_values(
        by="total_cost", ascending=False
    ).head(10)
    fig_models_cost = px.bar(
        month_data_models_cost,
        x="model",
        y="total_cost",
        title=f"Top 10 Total Cost by Model ({month})",
    )
    st.plotly_chart(fig_models_cost, use_container_width=True)

    # ---------------------------------
    # User Cost and Token Bar Plot
    # ---------------------------------
    # Group by user and sum both total_cost and total_tokens
    month_data_users = (
        month_data.groupby("user")[["total_cost", "total_tokens"]]
        .sum()
        .reset_index()
    )
    month_data_users = month_data_users.sort_values(by="total_cost", ascending=False)

    # Calculate totals for cost and tokens
    total_cost_sum = month_data_users["total_cost"].sum()
    total_tokens_sum = month_data_users["total_tokens"].sum()

    # Create the 'Total' row DataFrame
    total_row = pd.DataFrame(
        {
            "user": ["Total"],
            "total_cost": [total_cost_sum],
            "total_tokens": [total_tokens_sum],
        }
    )

    # Concatenate the original data with the 'Total' row
    month_data_users = pd.concat([month_data_users, total_row], ignore_index=True)

    # Plot only total cost for clarity, but keep tokens in the DataFrame
    fig_users = px.bar(
        month_data_users[
            month_data_users["user"] != "Total"
        ],  # Exclude Total row from plot for better scale
        x="user",
        y="total_cost",
        title=f"Total Cost by User ({month})",
    )
    st.plotly_chart(fig_users, use_container_width=True)

    # ---------------------------------
    # Collapsible DataFrames
    # ---------------------------------
    with st.expander("Show DataFrames"):
        st.subheader("Month Data")
        st.dataframe(month_data)
        st.subheader("Month Data Models Tokens")
        st.dataframe(month_data_models_tokens)
        st.subheader("Month Data Models Cost")
        st.dataframe(month_data_models_cost)
        st.subheader("Month Data Users")
        st.dataframe(month_data_users)


def main():
    st.title("Cost Tracker for Open WebUI")
    st.divider()

    st.page_link(
        "https://github.com/bgeneto/open-webui-cost-tracker/",
        label="GitHub Page",
        icon="🏠",
    )

    st.sidebar.title("⚙️ Options")

    st.info(
        "This Streamlit app processes and visualizes cost data from a JSON file. Select a JSON file below and a month to plot the data."
    )

    file = st.file_uploader("Upload a JSON file", type=["json"])

    if file is not None:
        data = load_data(file)
        if data is not None:
            processed_data = process_data(data)
            months = processed_data["month"].unique()
            month = st.sidebar.selectbox("Select a month", months)
            if st.button("Process Data"):
                plot_data(processed_data, month)
            if st.sidebar.button("Plot Data"):
                plot_data(processed_data, month)

if __name__ == "__main__":
    main()
