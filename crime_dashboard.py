
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="California Crime Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("chapter1log (1).csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    return df

df = load_data()

# Only keep per capita columns
per_capita_cols = [col for col in df.columns if '_per_100k' in col]
metric_names = {col: col.replace("_per_100k", "").replace("_", " ").title() for col in per_capita_cols}

# Sidebar filters
st.sidebar.title("Filters")
counties = st.sidebar.multiselect("Select County", options=df['County'].unique(), default=["Los Angeles County"])
selected_metric = st.sidebar.selectbox("Select Crime Metric", options=per_capita_cols, format_func=lambda x: metric_names[x])

# Filtered data
filtered_df = df[df['County'].isin(counties)]

# Plot
st.title("📊 California Crime per 100k Dashboard")
st.markdown("Compare standardized (per 100k) crime rates over time across California counties.")

st.subheader(f"{metric_names[selected_metric]} Over Time")
plot_data = filtered_df.groupby(['Date', 'County'])[selected_metric].mean().reset_index()
fig = px.line(plot_data, x='Date', y=selected_metric, color='County', title=metric_names[selected_metric])
st.plotly_chart(fig, use_container_width=True)
