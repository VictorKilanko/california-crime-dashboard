import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(layout="wide", page_title="California Crime Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("chapter1final.csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    rename_map = {
        "Violent_per_100k": "Violent Crime Rate",
        "Property_per_100k": "Property Crime Rate",
        "Homicide_per_100k": "Homicide Rate",
        "ForRape_per_100k": "Rape Rate",
        "FROBact_per_100k": "Robbery by Firearm Rate",
        "Robbery_per_100k": "Robbery Rate",
        "AggAssault_per_100k": "Aggravated Assault Rate",
        "Burglary_per_100k": "Burglary Rate",
        "FASSact_per_100k": "Assault by Firearm Rate",
        "LTtotal_per_100k": "Larceny Theft Rate",
        "VehicleTheft_per_100k": "Vehicle Theft Rate",
        "Arson_per_100k": "Arson Rate",
        "ViolentClr_per_100k": "Violent Crime Clearance Rate",
        "PropertyClr_per_100k": "Property Crime Clearance Rate"
    }
    return df.rename(columns=rename_map)

# --- Variables ---
crime_metrics = [
    "Violent Crime Rate", "Property Crime Rate", "Homicide Rate", "Rape Rate",
    "Robbery by Firearm Rate", "Robbery Rate", "Aggravated Assault Rate",
    "Burglary Rate", "Assault by Firearm Rate", "Larceny Theft Rate",
    "Vehicle Theft Rate", "Arson Rate", "Violent Crime Clearance Rate",
    "Property Crime Clearance Rate"
]

demographic_vars = [
    "Male Population", "Female Population", "White Population", "Black Population",
    "Asian Population", "Hispanic Population", "Foreign-Born Population",
    "Veteran Population", "Married Population", "Widowed Population",
    "Divorced Population", "Separated Population", "Never-Married Population",
    "Unemployed Population", "High School Graduates", "Bachelor's Degree Holders",
    "Graduate Degree Holders", "Children (0-17 years) Male", "Young Adults (18-24 years) Male",
    "Adults (25-44 years) Male", "Middle-aged Adults (45-64 years) Male", "Seniors (65+ years) Male",
    "Children (0-17 years) Female", "Young Adults (18-24 years) Female", "Adults (25-44 years) Female",
    "Middle-aged Adults (45-64 years) Female", "Seniors (65+ years) Female",
    "Male Veterans", "Female Veterans"
]

# --- Load Data ---
df = load_data()

# --- Sidebar ---
st.sidebar.title("Filters")
st.session_state["Page"] = st.sidebar.radio("Go to:", ["📈 Crime Trends", "📊 Explore Crime & Demographic Patterns"])

counties = st.sidebar.multiselect("Select County", options=df['County'].unique())
cities = st.sidebar.multiselect("Select City", options=df['City'].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", options=sorted(df['Year'].unique(), reverse=True))

filtered_df = df.copy()
if counties:
    filtered_df = filtered_df[filtered_df['County'].isin(counties)]
if cities:
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]
filtered_df = filtered_df[filtered_df['Year'] == selected_year]

# --- Crime Trends Page ---
if st.session_state["Page"] == "📈 Crime Trends":
    st.title("📈 California Crime Trends Over Time")

    st.markdown("Explore how crime rates and demographic metrics have changed over time in California counties and cities.")

    selected_metric = st.sidebar.selectbox("Select Crime Metric", options=crime_metrics)
    grouping_column = "City" if cities else "County"

    time_data = df.copy()
    if counties:
        time_data = time_data[time_data['County'].isin(counties)]
    if cities:
        time_data = time_data[time_data['City'].isin(cities)]

    plot_data = time_data.groupby(['Date', grouping_column])[selected_metric].mean().reset_index()
    fig = px.line(plot_data, x='Date', y=selected_metric, color=grouping_column,
                  title=f"{selected_metric} Over Time")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"This line chart shows how **{selected_metric}** has changed over time. Use it to identify trends, spikes, or declines in selected areas.")

    demo_metric = st.sidebar.selectbox("Add Demographic Line?", options=[None] + demographic_vars)
    if demo_metric:
        if demo_metric in time_data.columns:
            demo_data = time_data.groupby(['Date', grouping_column])[demo_metric].mean().reset_index()
            if not demo_data.empty:
                fig2 = px.line(demo_data, x='Date', y=demo_metric, color=grouping_column,
                               title=f"{demo_metric} Over Time")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown(f"This second chart overlays the demographic metric **{demo_metric}** over time. You can compare it visually with the crime trend.")
            else:
                st.warning("No data for selected demographic.")
        else:
            st.error(f"'{demo_metric}' not found.")

# --- Crime & Demographic Correlation Page ---
else:
    st.title("📊 Explore Crime & Demographic Patterns")

    st.markdown("""
    <div style='padding: 1em; background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; border-radius: 5px;'>
    <b>⚠️ Important:</b> Correlation does <i>not</i> imply causation. These plots show statistical relationships, but do not explain why the relationship exists.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ What does this page show?"):
        st.markdown("""
        - You can explore how different demographic groups relate to crime rates.
        - The scatter plots show whether there's a linear relationship (correlation) between the variables.
        - A correlation value closer to +1 or -1 means stronger association.
        - But correlation **does not** mean one causes the other.
        """)

    selected_crimes = st.multiselect("Select Crime Metric(s)", options=crime_metrics, default=["Violent Crime Rate"])
    demo_x = st.selectbox("Select Demographic Variable (X-axis)", options=demographic_vars, help="This will appear on the x-axis of your scatter plot.")

    show_corr = st.checkbox("Show correlation coefficient (r)", value=True)

    for crime in selected_crimes:
        st.markdown(f"### {crime} vs {demo_x}")
        st.markdown(f"This scatter plot compares **{demo_x}** to **{crime}** for the year {selected_year}. Each point represents a city or county.")

        plot_df = filtered_df[[demo_x, crime, 'City', 'County']].dropna()
        if not plot_df.empty:
            corr = plot_df[[demo_x, crime]].corr().iloc[0, 1]
            if show_corr:
                st.markdown(f"**Correlation (r): {corr:.2f}**")

            fig = px.scatter(
                plot_df, x=demo_x, y=crime, color='County',
                hover_data=['City'], trendline="ols",
                title=f"{crime} vs {demo_x} ({selected_year})"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for this combination.")

    # --- Clustering ---
    st.markdown("---")
    st.subheader("🔍 Clustering: Group Cities/Counties by Crime & Demographics")
    st.markdown("Use KMeans clustering to group locations with similar characteristics based on selected variables.")

    cluster_vars = st.multiselect("Select variables for clustering", options=demographic_vars + crime_metrics, default=["Violent Crime Rate", "Hispanic Population"])
    num_clusters = st.slider("Select number of clusters", 2, 6, 3)

    cluster_df = filtered_df[cluster_vars + ['City', 'County']].dropna()
    if not cluster_df.empty:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_df[cluster_vars])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(scaled)
        cluster_df['Cluster'] = kmeans.labels_

        fig3 = px.scatter(
            cluster_df, x=cluster_vars[0], y=cluster_vars[1], color='Cluster',
            hover_data=['City', 'County'], title="KMeans Clustering by Selected Variables"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f"This chart groups locations into **{num_clusters} clusters** based on the variables you've chosen. Similar profiles fall into the same cluster.")
    else:
        st.warning("Not enough data for clustering.")

    # --- Heatmap ---
    st.markdown("---")
    st.subheader("📌 Heatmap: Correlation Between Crimes and Demographics")
    st.markdown("This heatmap shows the strength of linear relationships between all selected crime and demographic variables.")

    heat_df = filtered_df[demographic_vars + crime_metrics].dropna()
    if not heat_df.empty:
        corr_matrix = heat_df.corr()
        fig4, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix.loc[crime_metrics, demographic_vars], annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig4)
    else:
        st.warning("Insufficient data to compute correlation matrix.")
