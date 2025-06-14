import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="California Crime Dashboard")

# --- Load and Prepare Data ---
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

df = load_data()

# --- Lists ---
crime_metrics = list(df.columns[df.columns.str.contains("Rate")])
demographic_vars = [col for col in df.columns if col not in crime_metrics + ['Date', 'Month', 'Year', 'County', 'City']]

# --- Sidebar ---
st.sidebar.title("Filters")
st.session_state["Page"] = st.sidebar.radio(
    "Go to:", 
    ["🏠 Page 1: Welcome", "📈 Page 2: Crime Trends", 
     "📉 Page 3: Demographic Context", "🔍 Page 4: Predict or Explain Crime"]
)

counties = st.sidebar.multiselect("Select County", options=df['County'].unique())
cities = st.sidebar.multiselect("Select City", options=df['City'].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", options=sorted(df['Year'].unique(), reverse=True))

# --- Apply Filters ---
filtered_df = df.copy()
if counties:
    filtered_df = filtered_df[filtered_df['County'].isin(counties)]
if cities:
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]
filtered_df = filtered_df[filtered_df['Year'] == selected_year]

# --- Page 1: Welcome ---
if st.session_state["Page"] == "🏠 Page 1: Welcome":
    st.title("🏠 Welcome to the California Crime Dashboard")
    st.markdown("""
    This interactive dashboard helps you explore the **relationships between crime and demographics** across California cities and counties.

    ---  

    ### 🔍 What You Can Do Here
    - 📈 **Crime Trends**: See how crime has evolved over time.
    - 📊 **Compare with Demographics**: Understand how income, education, and more relate to crime rates.
    - 🧠 **Run Predictive Models**: Try out machine learning to **predict crime rates** based on community factors.
    - 🎛️ **What-If Simulations**: Adjust demographics to see projected crime levels.

    ---

    ### 🧭 Getting Started
    1. Use the **sidebar** to navigate between pages.
    2. Apply filters by **county**, **city**, or **year** to focus your view.
    3. Dive deeper using the **predictive tools** or demographic visualizations.

    ---

    ⚠️ **Note**: This dashboard is for educational and exploratory purposes only. It does **not imply causation**.
    """)

# --- Page 2: Crime Trends ---
if st.session_state["Page"] == "📈 Page 2: Crime Trends":
    st.title("📈 California Crime Trends Over Time")
    selected_metric = st.sidebar.selectbox("Select Crime Metric", options=crime_metrics)
    demo_metric = st.sidebar.selectbox("Add Demographic Line?", options=[None] + demographic_vars)
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
    if demo_metric:
        try:
            demo_data = time_data.groupby(['Date', grouping_column])[demo_metric].mean().reset_index()
            fig2 = px.line(demo_data, x='Date', y=demo_metric, color=grouping_column,
                           title=f"{demo_metric} Over Time")
            st.plotly_chart(fig2, use_container_width=True)
        except:
            st.warning("Could not plot demographic variable.")

# --- Page 3: Demographic Context ---
if st.session_state["Page"] == "📉 Page 3: Demographic Context":
    st.title("📊 Compare Crime Trends by Demographic Groups")
    st.markdown("""
    Use this tool to explore how crime rates differ across different segments of the population.

    - Select a demographic variable (like % widowed or % below poverty).
    - The population is split into equal-sized groups based on that variable.
    - You’ll see how crime rates compare over time across those groups.

    This helps identify if certain demographic characteristics are associated with higher or lower crime trends.
    """)

    crime_var = st.selectbox("Select Crime Rate", options=crime_metrics, key="crime_var_page3")
    demo_var = st.selectbox("Select Demographic Variable", options=demographic_vars, key="demo_var_page3")
    year_filter = st.selectbox("Select Year", options=sorted(df['Year'].unique(), reverse=True), key="year_filter_page3")

    context_df = df[df['Year'] == year_filter][[crime_var, demo_var, 'County', 'City']].dropna(subset=[crime_var, demo_var])

    try:
        num_bins = st.slider("Number of Bins", 3, 8, 5, key="bin_slider_page3")
        context_df['DemoBin'] = pd.qcut(context_df[demo_var], q=num_bins, duplicates='drop').astype(str)
        binned = context_df.groupby('DemoBin')[crime_var].mean().reset_index()

        fig = px.bar(binned, x='DemoBin', y=crime_var,
                     labels={'DemoBin': f'{demo_var} Group', crime_var: f'Avg {crime_var}'},
                     title=f"Average {crime_var} by {demo_var} Groups ({year_filter})")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

    # Trend Over Time
    with st.expander("📈 Compare Trends Over Time", expanded=False):
        time_df = df[['Date', crime_var, demo_var, 'County', 'City']].dropna(subset=[crime_var, demo_var])
        try:
            time_df['QuantileGroup'] = pd.qcut(time_df[demo_var], q=4, 
                                               labels=["Low", "Mid-Low", "Mid-High", "High"], 
                                               duplicates='drop').astype(str)
            trend_df = time_df.groupby(['Date', 'QuantileGroup'])[crime_var].mean().reset_index()

            fig2 = px.line(trend_df, x='Date', y=crime_var, color='QuantileGroup',
                           title=f"{crime_var} Over Time by {demo_var} Quartile",
                           labels={'QuantileGroup': f'{demo_var} Group'})
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Cannot compute trend quartiles: {e}")


# --- Page 4: Predict or Explain Crime ---
if st.session_state["Page"] == "🔍 Page 4: Predict or Explain Crime":
    st.title("🔍 Predict or Explain Crime")

    # Selection inputs
    target = st.selectbox("🎯 Select Crime Variable to Predict", options=crime_metrics)
    predictor_options = demographic_vars
    predictors = st.multiselect("📊 Select Predictor Variables", options=predictor_options, default=['Median Household Income'])
    subset_city = st.selectbox("🏙️ Optional: Filter by City", options=["All"] + sorted(df['City'].dropna().unique().tolist()))
    model_type = st.radio("🧠 Choose Model Type", ["Linear Regression", "Random Forest"], horizontal=True)

    modeling_df = df.dropna(subset=[target] + predictors).copy()
    if subset_city != "All":
        modeling_df = modeling_df[modeling_df['City'] == subset_city]

    if len(predictors) > 0 and not modeling_df.empty:
        X = modeling_df[predictors]
        y = modeling_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Model Metrics Section ---
        st.subheader("📈 Model Performance Metrics")
        st.markdown("""
        **R² (R-squared)** indicates how well the model explains the variation in the crime variable. A value closer to 1 means better prediction.

        **MAE (Mean Absolute Error)** shows the average prediction error in actual units — lower is better.

        **RMSE (Root Mean Square Error)** penalizes larger errors more heavily than MAE — again, lower is better.
        """)
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        # --- Actual vs Predicted Chart ---
        fig_actual_vs_pred = px.scatter(x=y_test, y=y_pred,
                                        labels={"x": "Actual", "y": "Predicted"},
                                        title="Actual vs Predicted Crime Rate")
        st.plotly_chart(fig_actual_vs_pred)

        # --- Feature Importance Section ---
        st.subheader("📌 Feature Importance")
        st.markdown("""
        Feature importance tells us which variables contributed most to the model's predictions.
        Higher values mean the variable had a stronger influence on the outcome.
        """)
        if model_type == "Linear Regression":
            coef_df = pd.DataFrame({"Variable": predictors, "Importance": model.coef_})
        else:
            coef_df = pd.DataFrame({"Variable": predictors, "Importance": model.feature_importances_})

        fig_imp, ax = plt.subplots()
        sns.barplot(data=coef_df, x="Importance", y="Variable", palette="coolwarm", ax=ax)
        st.pyplot(fig_imp)

        # --- What-If Simulation ---
        st.subheader("🎛️ What-If Simulation")
        st.markdown("""
        Adjust the sliders below to simulate how changing predictor values affects the predicted crime rate.
        This helps understand the impact of specific variables.
        """)

        user_inputs = {}
        for var in predictors:
            min_val = float(X[var].min())
            max_val = float(X[var].max())
            mean_val = float(X[var].mean())
            step = (max_val - min_val) / 100 if (max_val - min_val) > 1 else 0.1
            user_inputs[var] = st.slider(f"{var}", min_val, max_val, mean_val, step=step)

        input_df = pd.DataFrame([user_inputs])
        prediction = model.predict(input_df)[0]
        st.success(f"📌 Predicted {target}: {prediction:.2f}")

        # --- SHAP Explanation ---
        with st.expander("🔎 SHAP Explanation"):
            st.markdown("""
            SHAP (SHapley Additive exPlanations) helps explain the contribution of each predictor to the individual prediction above.
            Positive values increase the prediction, negative values decrease it.
            """)
            try:
                explainer = shap.TreeExplainer(model) if model_type == "Random Forest" else shap.Explainer(model, X_train)
                shap_values = explainer(input_df)
                fig_shap, ax2 = plt.subplots()
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"SHAP error: {e}")

        # --- Correlation Matrix ---
        with st.expander("📉 Correlation Matrix"):
            st.markdown("""
            This shows how strongly each variable is related to one another. 
            Correlation values range from -1 (perfect inverse) to +1 (perfect direct relationship).
            """)
            corr = modeling_df[[target] + predictors].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)
    else:
        st.warning("Please select predictors and ensure data availability.")
