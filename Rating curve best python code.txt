import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rating Curve Generator",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Automatic Rating Curve Generator")

st.write(
    """
    Upload a CSV file containing **Stage** and **Discharge** data.
    The application will automatically generate and fit a rating curve.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Rating Curve Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


# ============================================================
# RATING CURVE FUNCTION
# Q = a(H - H0)^b
# ============================================================

def rating_curve(H, a, H0, b):

    return a * np.power(H - H0, b)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload a CSV file containing two columns: "
        "**Stage** and **Discharge**."
    )

    st.markdown(
        """
        ### Example CSV format

        | Stage | Discharge |
        |------:|----------:|
        | 0.20  | 1.5 |
        | 0.30  | 2.4 |
        | 0.40  | 3.8 |
        | 0.50  | 5.7 |
        | 0.60  | 8.2 |
        | 0.70  | 11.4 |

        Your CSV should contain:

        - **Stage** → water level / gauge height
        - **Discharge** → discharge corresponding to the stage
        """
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read CSV file: {e}")

    st.stop()


# ============================================================
# CHECK COLUMNS
# ============================================================

# Remove spaces from column names
df.columns = df.columns.str.strip()


if "Stage" not in df.columns or "Discharge" not in df.columns:

    st.error(
        "The CSV must contain columns named exactly "
        "**Stage** and **Discharge**."
    )

    st.write("Columns detected in your file:")

    st.write(list(df.columns))

    st.stop()


# ============================================================
# CONVERT DATA TO NUMERIC
# ============================================================

df["Stage"] = pd.to_numeric(df["Stage"], errors="coerce")
df["Discharge"] = pd.to_numeric(df["Discharge"], errors="coerce")


# Remove missing values
df = df.dropna(subset=["Stage", "Discharge"])


# Remove negative discharge
df = df[df["Discharge"] >= 0]


# Sort by stage
df = df.sort_values("Stage").reset_index(drop=True)


# ============================================================
# CHECK DATA
# ============================================================

if len(df) < 3:

    st.error(
        "At least 3 valid Stage–Discharge observations "
        "are required to generate a rating curve."
    )

    st.stop()


# ============================================================
# DISPLAY DATA
# ============================================================

st.subheader("📊 Uploaded Data")

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# BASIC INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Number of observations",
        len(df)
    )

with col2:
    st.metric(
        "Minimum Stage",
        f"{df['Stage'].min():.3f}"
    )

with col3:
    st.metric(
        "Maximum Stage",
        f"{df['Stage'].max():.3f}"
    )


# ============================================================
# FITTING THE RATING CURVE
# ============================================================


discharge = df["Stage"].values
stage = df["Discharge"].values


# Initial estimate of H0
# Slightly below the minimum observed stage
stage_range = stage.max() - stage.min()

if stage_range == 0:

    st.error("Stage values must not all be the same.")

    st.stop()


initial_H0 = stage.min() - 0.05 * stage_range


# Initial estimates for a and b
initial_a = max(
    discharge.max() /
    max((stage.max() - initial_H0) ** 2, 1e-6),
    0.001
)

initial_b = 2.0


try:

    # Bounds for parameters
    lower_bounds = [
        0.000001,
        stage.min() - stage_range,
        0.1
    ]

    upper_bounds = [
        np.inf,
        stage.min() - 0.000001,
        10
    ]

    parameters, covariance = curve_fit(
        rating_curve,
        stage,
        discharge,
        p0=[
            initial_a,
            initial_H0,
            initial_b
        ],
        bounds=(
            lower_bounds,
            upper_bounds
        ),
        maxfev=50000
    )


    a, H0, b = parameters


except Exception as e:

    st.error(
        f"Unable to fit the rating curve automatically: {e}"
    )

    st.stop()


# ============================================================
# CALCULATE PREDICTED DISCHARGE
# ============================================================

predicted = rating_curve(
    stage,
    a,
    H0,
    b
)


# ============================================================
# STATISTICS
# ============================================================

residuals = discharge - predicted

SSE = np.sum(residuals ** 2)

SST = np.sum(
    (discharge - np.mean(discharge)) ** 2
)

if SST != 0:

    R2 = 1 - SSE / SST

else:

    R2 = np.nan


RMSE = np.sqrt(
    np.mean(residuals ** 2)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

st.subheader("📐 Rating Curve Equation")

st.success(
    f"**Q = {a:.4f} × (H − {H0:.4f})^{b:.4f}**"
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Coefficient a",
        f"{a:.4f}"
    )

with col2:

    st.metric(
        "Exponent b",
        f"{b:.4f}"
    )

with col3:

    st.metric(
        "Zero-flow stage H₀",
        f"{H0:.4f}"
    )

with col4:

    st.metric(
        "R²",
        f"{R2:.4f}"
    )


st.write(
    f"**RMSE:** {RMSE:.4f}"
)


# ============================================================
# CREATE SMOOTH CURVE
# ============================================================

curve_stage = np.linspace(
    max(H0 + 0.0001, stage.min()),
    stage.max(),
    300
)

curve_discharge = rating_curve(
    curve_stage,
    a,
    H0,
    b
)


# ============================================================
# PLOT
# ============================================================

st.subheader("📈 Stage–Discharge Rating Curve")


fig, ax = plt.subplots(
    figsize=(10, 6)
)


# Measured data
ax.scatter(
    stage,
    discharge,
    label="Observed Data",
    s=60
)


# Fitted curve
ax.plot(
    curve_stage,
    curve_discharge,
    linewidth=2,
    label="Fitted Rating Curve"
)


ax.set_xlabel(
    "Discharge (Q)"
)

ax.set_ylabel(
    "Stage (H)"
)

ax.set_title(
    "Stage–Discharge Rating Curve"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


st.pyplot(fig)


# ============================================================
# PREDICT DISCHARGE
# ============================================================

st.subheader("🔢 Estimate Discharge from Stage")

user_stage = st.number_input(
    "Enter Stage",
    min_value=float(stage.min()),
    max_value=float(stage.max()),
    value=float(stage.mean()),
    step=0.01
)


if user_stage <= H0:

    st.warning(
        "The entered stage is at or below the estimated "
        "zero-flow stage H₀."
    )

else:

    estimated_discharge = rating_curve(
        user_stage,
        a,
        H0,
        b
    )

    st.success(
        f"Estimated Discharge = "
        f"**{estimated_discharge:.3f}**"
    )


# ============================================================
# CREATE RATING TABLE
# ============================================================

rating_table = pd.DataFrame({

    "Stage": curve_stage,

    "Estimated_Discharge": curve_discharge

})


# ============================================================
# DOWNLOAD RATING CURVE
# ============================================================

st.subheader("⬇️ Download Rating Curve")


csv_output = rating_table.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Rating Curve CSV",
    data=csv_output,
    file_name="rating_curve.csv",
    mime="text/csv"
)


# ============================================================
# OBSERVED VS PREDICTED
# ============================================================

st.subheader("📋 Observed vs Predicted Discharge")


comparison = pd.DataFrame({

    "Stage": stage,

    "Observed_Discharge": discharge,

    "Predicted_Discharge": predicted,

    "Residual": residuals

})


st.dataframe(
    comparison,
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Automatic Stage–Discharge Rating Curve Generator"
)