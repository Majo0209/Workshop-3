import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sqlalchemy import create_engine
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
import joblib, os

pio.templates.default = "plotly_white"

default_font = dict(color="black", size=13)
axis_style = dict(
    title_font=dict(size=15, color="black"),
    tickfont=dict(size=12, color="black"),
    showgrid=True,
    zeroline=False,
    linecolor="lightgray"
)

# --------------------------------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# --------------------------------------------------------------
st.set_page_config(
    page_title="Happiness Model Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Happiness Model Dashboard")
st.caption("Visual evaluation and KPIs of the Ridge Regression model (Happiness Prediction)")

# --------------------------------------------------------------
# CONEXIÓN Y CARGA DE DATOS
# --------------------------------------------------------------
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
DW_SCHEMA = "happiness_dw"

@st.cache_data
def load_data():
    try:
        engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{DW_SCHEMA}"
        )
        df = pd.read_sql("SELECT * FROM predicciones_happiness;", engine)
        return df
    except Exception:
        df = pd.read_csv("data/processed/happiness_predictions_Ridge.csv")
        return df

df = load_data()
df["year"] = df["year"].astype(int)

# --------------------------------------------------------------
# SIDEBAR - FILTROS
# --------------------------------------------------------------
st.sidebar.header("Visualization Filters")

years = sorted(df["year"].unique())
selected_years = st.sidebar.multiselect("Select year(s):", years, default=years)
set_sel = st.sidebar.radio("Select dataset:", ("All", "Training", "Test"))

df_filt = df[df["year"].isin(selected_years)]
if set_sel == "Training":
    df_filt = df_filt[df_filt["is_train"] == 1]
elif set_sel == "Test":
    df_filt = df_filt[df_filt["is_test"] == 1]

total = len(df_filt)
train_count = len(df[df["is_train"] == 1])
test_count = len(df[df["is_test"] == 1])

st.sidebar.markdown("### Records Summary")
if set_sel == "All":
    st.sidebar.info(f"Total records: **{total:,}** (Train={train_count:,} | Test={test_count:,})")
elif set_sel == "Training":
    st.sidebar.info(f"Training records: **{total:,}**")
else:
    st.sidebar.info(f"Test records: **{total:,}**")

if not selected_years:
    st.warning("Please select at least one year in the sidebar to view the visualizations.")
    st.stop()

# --------------------------------------------------------------
# CÁLCULO DE MÉTRICAS
# --------------------------------------------------------------
r2 = r2_score(df_filt["score_real"], df_filt["score_predicho"])
rmse = np.sqrt(mean_squared_error(df_filt["score_real"], df_filt["score_predicho"]))
mae = mean_absolute_error(df_filt["score_real"], df_filt["score_predicho"])
mape = mean_absolute_percentage_error(df_filt["score_real"], df_filt["score_predicho"]) * 100
bias = np.mean(df_filt["score_predicho"] - df_filt["score_real"])

# --------------------------------------------------------------
# VISUALIZACIÓN DE INDICADORES
# --------------------------------------------------------------
st.markdown("### Model Performance Indicators")
col_kpis = st.columns(5)
for col, (label, value) in zip(
    col_kpis,
    [
        ("R²", f"{r2:.3f}"),
        ("RMSE", f"{rmse:.3f}"),
        ("MAE", f"{mae:.3f}"),
        ("MAPE (%)", f"{mape:.2f}"),
        ("Bias", f"{bias:.3f}")
    ]
):
    col.metric(label, value)

# --------------------------------------------------------------
# MAPA MUNDIAL DE FELICIDAD
# --------------------------------------------------------------
st.markdown("### Predicted Happiness by Country")
try:
    fig_map = px.choropleth(
        df_filt,
        locations="country",
        locationmode="country names",
        color="score_predicho",
        hover_name="country",
        color_continuous_scale="Viridis",
        title=f"Predicted Happiness Score ({set_sel} dataset)",
    )
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        font=default_font,
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(color="black")
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Predicted Score", font=dict(color="black", size=13)),
            tickfont=dict(color="black", size=12)
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render map: {e}")

st.markdown("")

# --------------------------------------------------------------
# VISUALIZACIONES PRINCIPALES
# --------------------------------------------------------------
col1, col2 = st.columns(2, gap="medium")

# --- 1. Real vs Predicho
with col1:
    st.subheader("Real vs Predicted Score (colored by year)")
    df_filt["year_str"] = df_filt["year"].astype(str)
    fig1 = px.scatter(
        df_filt, x="score_real", y="score_predicho", color="year_str",
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"score_real": "Real Score", "score_predicho": "Predicted Score", "year_str": "Year"},
        hover_data={"score_real": ":.3f", "score_predicho": ":.3f"},
        opacity=0.8, height=420
    )
    fig1.add_shape(
        type="line",
        x0=df_filt["score_real"].min(), y0=df_filt["score_real"].min(),
        x1=df_filt["score_real"].max(), y1=df_filt["score_real"].max(),
        line=dict(color="red", dash="dash", width=2)
    )
    fig1.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font=default_font,
        legend_title_text="Year",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="lightgray", borderwidth=1,
            font=dict(color="black")
        ),
        xaxis=axis_style, yaxis=axis_style, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- 2. Promedio anual
with col2:
    st.subheader("Annual Average of Scores")
    df_year = df_filt.groupby("year")[["score_real", "score_predicho"]].mean().reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_year["year"], y=df_year["score_real"],
        mode="lines+markers", name="Real",
        line=dict(color="#1f77b4", width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=df_year["year"], y=df_year["score_predicho"],
        mode="lines+markers", name="Predicted",
        line=dict(color="#ff0ea3", width=2)
    ))
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font=default_font,
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="lightgray", borderwidth=1,
            font=dict(color="black")
        ),
        xaxis=axis_style, yaxis=axis_style,
        xaxis_title="Year", yaxis_title="Average Score",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- 3. Barras Real vs Predicho por año
with col1:
    st.subheader("Average Score per Year (Real vs Predicted)")
    df_year_melt = df_year.melt(
        id_vars="year", value_vars=["score_real", "score_predicho"],
        var_name="Type", value_name="Score"
    )
    fig3 = px.bar(
        df_year_melt, x="year", y="Score", color="Type",
        barmode="group",
        color_discrete_sequence=["#efef92", "#ecbbf7"],
        labels={"year": "Year", "Score": "Average Score"},
        hover_data={"Score": ":.3f"}, height=420
    )
    fig3.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font=default_font,
        legend_title_text="Type",
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="lightgray", borderwidth=1,
            font=dict(color="black")
        ),
        xaxis=axis_style, yaxis=axis_style, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- 4. Distribución del error
with col2:
    st.subheader("Error Distribution (Predicted - Real)")
    df_filt["error"] = df_filt["score_predicho"] - df_filt["score_real"]
    fig4 = px.histogram(
        df_filt, x="error", nbins=25, marginal="box",
        color_discrete_sequence=["#e89494"],
        labels={"error": "Error (Predicted - Real)"},
        hover_data={"error": ":.3f"}, height=420
    )
    fig4.add_vline(x=0, line_dash="dash", line_color="red")
    fig4.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font=default_font,
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="lightgray", borderwidth=1,
            font=dict(color="black")
        ),
        xaxis=axis_style, yaxis=axis_style,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig4, use_container_width=True)

# --- 5. Importancia de variables
with col1:
    st.subheader("Feature Importance (Ridge Coefficients)")
    try:
        model_path = "models/Best_Ridge_Model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            # Detectar nombres automáticamente si están disponibles
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
            else:
                # Fallback a nombres genéricos si no existen
                feature_names = [f"Feature_{i}" for i in range(len(model.coef_))]

            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", key=abs, ascending=False)

            fig5 = px.bar(
                coef_df, x="Coefficient", y="Feature",
                orientation="h", color="Coefficient",
                color_continuous_scale="RdBu",
                labels={"Coefficient": "Model Weight"},
                height=420
            )
            fig5.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                font=default_font, coloraxis_showscale=False,
                xaxis=axis_style, yaxis=axis_style,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("Ridge model (.pkl) not found in the models/ folder.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
