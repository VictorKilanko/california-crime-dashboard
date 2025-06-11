
import streamlit as st
import pandas as pd
import plotly.express as px

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
    
    df = df.rename(columns=rename_map)
    return df

# Load and prep data
df = load_data()

# Identify crime metric columns
crime_metrics = [
    "Violent Crime Rate", "Property Crime Rate", "Homicide Rate", "Rape Rate", 
    "Robbery by Firearm Rate", "Robbery Rate", "Aggravated Assault Rate", 
    "Burglary Rate", "Assault by Firearm Rate", "Larceny Theft Rate", 
    "Vehicle Theft Rate", "Arson Rate", "Violent Crime Clearance Rate", 
    "Property Crime Clearance Rate"
]

# Sidebar filters
st.sidebar.title("Filters")
counties = st.sidebar.multiselect("Select County", options=df['County'].unique())
cities = st.sidebar.multiselect("Select City", options=df['City'].dropna().unique())
selected_metric = st.sidebar.selectbox("Select Crime Metric", options=per_capita_cols, format_func=lambda x: metric_names[x])

# Apply filters
filtered_df = df.copy()
if counties:
    filtered_df = filtered_df[filtered_df['County'].isin(counties)]
if cities:
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]


# Plot
st.title("📊 California Crime Dashboard (per 100k)")
st.markdown("Visualizing crime and clearance rates across cities and counties in California. Select metrics and filters from the sidebar.")

st.subheader(f"{metric_names[selected_metric]} Over Time")

grouping_column = "City" if cities else "County"
plot_data = filtered_df.groupby(['Date', grouping_column])[selected_metric].mean().reset_index()

fig = px.line(plot_data, x='Date', y=selected_metric, color=grouping_column, title=metric_names[selected_metric])
st.plotly_chart(fig, use_container_width=True)
