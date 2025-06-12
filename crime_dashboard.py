import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

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
    ["🏠 Welcome", "📈 Crime Trends", 
     "📉 Demographic Crime Context", "🔍 Predict or Explain Crime"]
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


# --- Landing Page ---
if st.session_state["Page"] == "🏠 Welcome":
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


# ==========================
# 📈 PAGE 1: Crime Trends
# ==========================
if st.session_state["Page"] == "📈 Crime Trends":
    st.title("📈 California Crime Trends Over Time")

    with st.expander("🧭 How to Use This Page", expanded=True):
        st.markdown("""
        This page shows **how crime rates and population factors change over time**.

        1. Use the **sidebar** to select a crime metric and location(s).
        2. View the crime rate trend over time.
        3. *(Optional)* Add a **demographic variable** to compare against crime.

        Example: How has *Vehicle Theft Rate* changed over time in Alameda County?
        """)

    selected_metric = st.sidebar.selectbox("Select Crime Metric", options=crime_metrics)
    demo_metric = st.sidebar.selectbox("Add Demographic Line?", options=[None] + demographic_vars)

    grouping_column = "City" if cities else "County"
    time_data = df.copy()
    if counties:
        time_data = time_data[time_data['County'].isin(counties)]
    if cities:
        time_data = time_data[time_data['City'].isin(cities)]

    # Crime line plot
    plot_data = time_data.groupby(['Date', grouping_column])[selected_metric].mean().reset_index()
    fig = px.line(plot_data, x='Date', y=selected_metric, color=grouping_column,
                  title=f"{selected_metric} Over Time")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"📊 **{selected_metric}** over time. Look for spikes, drops, or long-term patterns.")

    # Optional demographic line
    if demo_metric:
        if demo_metric in time_data.columns:
            demo_data = time_data.groupby(['Date', grouping_column])[demo_metric].mean().reset_index()
            if not demo_data.empty:
                fig2 = px.line(demo_data, x='Date', y=demo_metric, color=grouping_column,
                               title=f"{demo_metric} Over Time")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown(f"📈 **{demo_metric}** trend — useful for side-by-side comparisons.")
            else:
                st.warning("No data available for the selected demographic.")
        else:
            st.error(f"'{demo_metric}' not found in dataset.")

# ==========================
# 📉 PAGE 2: Demographic Context
# ==========================
if st.session_state["Page"] == "📉 Demographic Crime Context":
    st.title("📉 Crime Patterns Across Demographic Contexts")

    with st.expander("🧭 How to Use This Page", expanded=True):
        st.markdown("""
        This page helps explore **how crime varies across places with different demographics**.

        1. Pick a crime rate and a demographic variable.
        2. The chart groups cities/counties based on the demographic value.
        3. *(Optional)* Expand below to compare **trends over time** by demographic quartiles.

        > This doesn't mean people from a group commit more crime — it shows **patterns across places**.
        """)

    crime_var = st.selectbox("Select Crime Rate", options=crime_metrics)
    demo_var = st.selectbox("Select Demographic Variable", options=demographic_vars)
    year_filter = st.selectbox("Select Year", options=sorted(df['Year'].unique(), reverse=True))

    context_df = df[df['Year'] == year_filter][[crime_var, demo_var, 'County', 'City']].dropna()

    try:
        num_bins = st.slider("Number of Bins (for Demographic Groups)", 3, 8, 5)
        context_df['DemoBin'] = pd.qcut(context_df[demo_var], q=num_bins, duplicates='drop')
        binned = context_df.groupby('DemoBin')[crime_var].mean().reset_index()

        fig = px.bar(
            binned, x='DemoBin', y=crime_var,
            title=f"Average {crime_var} by {demo_var} Groupings ({year_filter})",
            labels={"DemoBin": f"{demo_var} Group", crime_var: f"Avg {crime_var}"},
            color_discrete_sequence=["#1f77b4"]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"📊 Places are grouped by **{demo_var}** levels. You can see how average **{crime_var}** changes across those groups.")
    except Exception as e:
        st.error(f"Could not create demographic bins: {e}")

    # --- Optional Trend Comparison ---
    with st.expander("📈 Optional: Compare Time Trends for High vs. Low Demographic Groups"):
        time_df = df[['Date', crime_var, demo_var, 'County', 'City']].dropna()
        try:
            time_df['QuantileGroup'] = pd.qcut(time_df[demo_var], q=4, labels=["Low", "Mid-Low", "Mid-High", "High"], duplicates='drop')
            trend_df = time_df.groupby(['Date', 'QuantileGroup'])[crime_var].mean().reset_index()

            fig2 = px.line(
                trend_df, x='Date', y=crime_var, color='QuantileGroup',
                title=f"{crime_var} Over Time by {demo_var} Quartile",
                labels={"QuantileGroup": f"{demo_var} Group", crime_var: crime_var}
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("📈 This trend compares **crime over time** across demographic groups (quartiles).")
        except:
            st.warning("Unable to compute quartiles for selected demographic.")




# --- Page 4: Predict or Explain Crime (Enhanced with Interpretability) ---
if st.session_state["Page"] == "🔍 Predict or Explain Crime":
    import shap
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import seaborn as sns

    st.title("🔍 Predict or Explain Crime with Data")

    with st.expander("🧭 How to Use", expanded=True):
        st.markdown("""
        1. Choose a **crime variable** to explain or predict.
        2. Select **predictor variables** like demographics or income.
        3. Pick a **model type**: linear or non-linear (random forest).
        4. Optionally filter by **city**.
        5. View prediction results, importance of features, and run **What-If simulations**.
        6. Use **SHAP** to understand how individual variables affect predictions.

        > This is an **exploration tool**, not for causal inference or forecasting.
        """)

    target = st.selectbox("🎯 Select Crime Variable to Predict", options=crime_metrics)

    predictor_options = [
        'Hispanic Population', 'White Population', 'Black Population', 'Asian Population',
        'Married Population', 'Never-Married Population', 'Median Household Income',
        'Unemployed Population', 'Bachelor\'s Degree Holders', 'Graduate Degree Holders',
        'Homeowners Population', 'Population Below Poverty'
    ]
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

        # 📈 Model Results
        st.subheader("📈 Model Results")
        st.markdown("""
        📌 **R² Score** (explained variance):  
        - Measures how well the model explains variation in the crime rate.
        - **1.00** = perfect model, **0.00** = no predictive power.
        """)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        st.metric("R² Score", f"{r2:.2f}")
        st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")

        fig_actual_vs_pred = px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "Actual Crime Rate", "y": "Predicted Crime Rate"},
            title="📍 Actual vs Predicted Crime Rate"
        )
        st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

        # 🔍 Feature Importance
        st.subheader("📊 Feature Importance")
        st.markdown("""
        📌 **What this shows:**  
        - The **relative impact** each variable had on the predictions.  
        - Positive values ↑ crime; negative ↓ crime (in linear regression).  
        """)

        if model_type == "Linear Regression":
            coef_df = pd.DataFrame({
                "Variable": predictors,
                "Importance": model.coef_
            }).sort_values(by="Importance", key=abs, ascending=False)
        else:
            coef_df = pd.DataFrame({
                "Variable": predictors,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

        fig_imp, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=coef_df, x="Importance", y="Variable", palette="coolwarm", ax=ax)
        st.pyplot(fig_imp)

        # 🎛️ What-If Simulation
        st.subheader("🎛️ Simulate 'What If' Scenario")
        st.markdown("""
        📌 Change predictor values to simulate their effect on predicted crime.  
        Great for exploring *hypotheticals* like:  
        - What if income increases?
        - What if unemployment drops?
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
        st.success(f"📌 Predicted **{target}**: `{prediction:.2f}` based on your input values.")

        # 🔎 SHAP Explanations
        with st.expander("🔎 SHAP Explanation for This Prediction"):
            st.markdown("""
            📌 **SHAP** explains how each input variable contributed to this prediction:  
            - Positive SHAP value: pushed prediction higher.  
            - Negative SHAP value: pulled prediction lower.  
            """)

            with st.spinner("Calculating SHAP values..."):
                if model_type == "Random Forest":
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_train)

                shap_values = explainer(input_df)

                fig_shap, ax2 = plt.subplots()
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig_shap)

        # 🧪 Correlation Matrix
        with st.expander("📉 Correlation Matrix"):
            st.markdown("""
            📌 **Correlations** show how strongly variables relate:  
            - +1 = Strong positive  
            - -1 = Strong negative  
            - 0 = No relationship  
            """)
            corr_cols = [target] + predictors
            corr = modeling_df[corr_cols].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)

    else:
        st.warning("Please select at least one predictor and ensure data is available.")
