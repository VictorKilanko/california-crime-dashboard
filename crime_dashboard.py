import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="California Crime Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("chapter1final.csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

    # Rename crime columns
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

df = load_data()

# Columns for crime metrics
per_capita_cols = [
    "Violent Crime Rate", "Property Crime Rate", "Homicide Rate", "Rape Rate",
    "Robbery by Firearm Rate", "Robbery Rate", "Aggravated Assault Rate",
    "Burglary Rate", "Assault by Firearm Rate", "Larceny Theft Rate",
    "Vehicle Theft Rate", "Arson Rate", "Violent Crime Clearance Rate",
    "Property Crime Clearance Rate"
]

metric_names = {col: col for col in per_capita_cols}

# Demographic variables to choose from
demographic_vars = [
    'Male Population', 'Female Population', 'White Population', 'Black Population',
    'Asian Population', 'Hispanic Population', 'Foreign-Born Population',
    'Veteran Population', 'Married Population', 'Widowed Population',
    'Divorced Population', 'Separated Population', 'Never-Married Population',
    'Unemployed Population', 'High School Graduates', "Bachelor's Degree Holders",
    'Graduate Degree Holders', 'Children (0-17 years) Male', 'Young Adults (18-24 years) Male',
    'Adults (25-44 years) Male', 'Middle-aged Adults (45-64 years) Male', 'Seniors (65+ years) Male',
    'Children (0-17 years) Female', 'Young Adults (18-24 years) Female',
    'Adults (25-44 years) Female', 'Middle-aged Adults (45-64 years) Female',
    'Seniors (65+ years) Female', 'Male Veterans', 'Female Veterans'
]

# Sidebar filters
st.sidebar.title("Filters")
counties = st.sidebar.multiselect("Select County", options=df['County'].dropna().unique())
cities = st.sidebar.multiselect("Select City", options=df['City'].dropna().unique())
selected_metric = st.sidebar.selectbox("Select Crime Metric", options=per_capita_cols, format_func=lambda x: metric_names[x])
selected_demo = st.sidebar.selectbox("Overlay Demographic Variable (optional)", options=["None"] + demographic_vars)

# Filter dataset
filtered_df = df.copy()
if counties:
    filtered_df = filtered_df[filtered_df['County'].isin(counties)]
if cities:
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]

# Check if there's data
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    grouping_column = "City" if cities else "County"

    plot_data = filtered_df.groupby(['Date', grouping_column]).agg({selected_metric: 'mean'})
    
    if selected_demo != "None":
        plot_data[selected_demo] = filtered_df.groupby(['Date', grouping_column])[selected_demo].mean()

    plot_data = plot_data.reset_index()

    st.title("📊 California Crime Dashboard (per 100k)")
    st.markdown("Explore crime trends alongside demographics across counties and cities.")

    st.subheader(f"{metric_names[selected_metric]} Over Time")

    # Plot
    fig = go.Figure()

    for group in plot_data[grouping_column].unique():
        df_group = plot_data[plot_data[grouping_column] == group]

        # Crime line
        fig.add_trace(go.Scatter(
            x=df_group['Date'], y=df_group[selected_metric], mode='lines',
            name=f"{group} - {selected_metric}", yaxis='y1'
        ))

        # Demographic line (secondary y-axis)
        if selected_demo != "None":
            fig.add_trace(go.Scatter(
                x=df_group['Date'], y=df_group[selected_demo], mode='lines',
                name=f"{group} - {selected_demo}", yaxis='y2', line=dict(dash='dot')
            ))

    # Layout settings
    layout = {
        "title": f"{metric_names[selected_metric]} Over Time",
        "xaxis": {"title": "Date"},
        "yaxis": {"title": selected_metric},
        "legend": {"orientation": "h"},
    }

    if selected_demo != "None":
        layout["yaxis2"] = {
            "title": selected_demo,
            "overlaying": "y",
            "side": "right",
            "showgrid": False
        }

    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)
