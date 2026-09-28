"""
🛒 Customer Purchase Prediction — Haveli Premium Edition (v4)
Author: <A.R.KHAN>
Theme: Luxury dark — midnight, gold, ivory (Rajasthani palace aesthetic)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Haveli Premium — Customer Purchase Prediction",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# LOAD CUSTOM CSS
# ------------------------------------------------------------------
def load_css(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# ------------------------------------------------------------------
# GLOBAL SAFETY CSS — re-applies premium colors after every rerun
# ------------------------------------------------------------------
st.markdown("""
<style>
    section.main .stMarkdown p,
    section.main .stMarkdown span,
    section.main .stMarkdown li,
    section.main .stMarkdown div { color: #F5EFE0 !important; }

    h1, h2, h3, h4, h5, h6 { color: #E8B84B !important; }

    label, .stNumberInput label, .stTextInput label, .stSelectbox label,
    .stFileUploader label, .stRadio label {
        color: #C9A34E !important;
        font-weight: 700 !important;
        letter-spacing: 0.6px !important;
        text-transform: uppercase;
        font-size: 0.86rem !important;
    }

    [data-testid="stCaptionContainer"], .stCaption, small {
        color: #C9B896 !important;
        opacity: 1 !important;
    }

    .stTabs [data-baseweb="tab"] p {
        color: #C9B896 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] p { color: #E8B84B !important; }

    .stDataFrame, .stDataFrame * { color: #F5EFE0 !important; }

    /* Sidebar text stays ivory */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] div { color: #F5EFE0 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# HERO HEADER
# ------------------------------------------------------------------
st.markdown("""
<div class="haveli-header">
    <div class="diya"></div>
    <h1>👑 हवेली — Customer Purchase Prediction</h1>
    <p>Where ancient wisdom meets modern intelligence</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression":  ("models/logistic_model.pkl",     True),
    "Decision Tree":        ("models/decision_tree_model.pkl", False),
    "Random Forest":        ("models/random_forest_model.pkl", False),
    "K-Nearest Neighbors":  ("models/knn_model.pkl",           True),
}

@st.cache_resource
def load_available_models():
    loaded = {}
    for name, (path, needs_scale) in MODEL_FILES.items():
        if not os.path.exists(path):
            continue
        if os.path.getsize(path) == 0:
            st.sidebar.warning(f"⚠️ {name}: file is empty")
            continue
        try:
            loaded[name] = {"model": joblib.load(path), "needs_scale": needs_scale}
        except Exception as e:
            st.sidebar.warning(f"⚠️ {name}: load failed ({type(e).__name__})")
    return loaded

@st.cache_resource
def load_scaler():
    path = "models/scaler.pkl"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

available_models = load_available_models()
scaler = load_scaler()

if not available_models:
    st.error("❌ No models found in `models/`. Run the notebook first.")
    st.stop()

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    selected_model_name = st.selectbox(
        "Choose a Model",
        list(available_models.keys()),
        help="Each model was trained on the same data — try them all!",
    )
    st.markdown("---")

    st.markdown("## 📖 About")
    st.markdown(
        "This app predicts whether a customer will purchase a product "
        "based on their **Age** and **Estimated Salary**."
    )
    st.markdown("---")

    st.markdown("## 🎭 Buyer Personas")
    st.markdown(
        "- 🌱 **Young Explorer** — Under 30, low-mid salary\n"
        "- 💼 **Rising Professional** — 30-45, mid salary\n"
        "- 🏛️ **Prosperous Elder** — 45+, high salary\n"
        "- 🧭 **Wanderer** — everything else"
    )
    st.markdown("---")
    st.caption("Crafted with ❤️ using Streamlit & Scikit-learn")

# ------------------------------------------------------------------
# PERSONA LOGIC
# ------------------------------------------------------------------
def get_persona(age: int, salary: int) -> str:
    if age < 30 and salary < 60000:
        return "🌱 Young Explorer"
    if 30 <= age < 45 and salary < 100000:
        return "💼 Rising Professional"
    if age >= 45 and salary >= 80000:
        return "🏛️ Prosperous Elder"
    return "🧭 Wanderer"

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_predict, tab_batch, tab_insights, tab_history = st.tabs(
    ["🔮 Predict", "📂 Batch Predict", "📊 Insights", "📜 History"]
)

if "history" not in st.session_state:
    st.session_state.history = []

# ==================================================================
# TAB 1 — PREDICT
# ==================================================================
with tab_predict:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(
            '<div class="haveli-card">'
            '<h3>🧾 Customer Details</h3>'
            '</div>',
            unsafe_allow_html=True,
        )
        age    = st.number_input("Age (आयु)", min_value=18, max_value=100, value=30, step=1)
        salary = st.number_input(
            "Estimated Salary (अनुमानित वेतन)",
            min_value=1000, max_value=300000, value=60000, step=1000,
        )
        predict_btn = st.button("✨ Predict Purchase", use_container_width=True)

        persona = get_persona(age, salary)
        st.markdown(
            f'<div class="haveli-card" style="text-align:center;">'
            f'<h3 style="margin-bottom:10px;">Customer Persona</h3>'
            f'<span class="persona-badge">{persona}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        if predict_btn:
            info  = available_models[selected_model_name]
            model = info["model"]
            data  = pd.DataFrame({"Age": [age], "EstimatedSalary": [salary]})

            if info["needs_scale"]:
                if scaler is None:
                    st.error("Scaler missing — needed for this model.")
                    st.stop()
                data_ready = scaler.transform(data)
            else:
                data_ready = data

            proba      = float(model.predict_proba(data_ready)[0, 1])
            pred       = int(proba >= 0.5)
            confidence = max(proba, 1 - proba) * 100

            if pred == 1:
                st.markdown(
                    f'<div class="result-buy">'
                    f'✅ Customer is likely to <b>BUY</b><br>'
                    f'Probability of purchase: <b>{proba*100:.2f}%</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-nobuy">'
                    f'❌ Customer is unlikely to <b>BUY</b><br>'
                    f'Probability of purchase: <b>{proba*100:.2f}%</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Probability gauge heading
            st.markdown(
                '<h4 style="color:#E8B84B; margin-top:24px; '
                'letter-spacing:0.6px;">🎯 Purchase Probability</h4>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="prob-track">'
                f'<div class="prob-fill" style="width:{proba*100:.1f}%;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#C9B896; font-weight:600; margin-top:12px; '
                f'letter-spacing:0.5px;">'
                f'Confidence: <b style="color:#E8B84B;">{confidence:.2f}%</b> '
                f'&nbsp;•&nbsp; Model: '
                f'<b style="color:#E8B84B;">{selected_model_name}</b>'
                f'</p>',
                unsafe_allow_html=True,
            )

            st.session_state.history.append({
                "Time":        datetime.now().strftime("%H:%M:%S"),
                "Age":         age,
                "Salary":      salary,
                "Model":       selected_model_name,
                "Prediction":  "Buy" if pred == 1 else "No Buy",
                "Probability": round(proba, 4),
                "Persona":     persona,
            })
        else:
            st.markdown(
                '<div class="haveli-card" style="text-align:center; padding:60px;">'
                '<h3 style="color:#E8B84B;">🏮 Awaiting your input...</h3>'
                '<p style="color:#C9B896;">Enter customer details and press '
                '<b style="color:#E8B84B;">Predict Purchase</b>.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

# ==================================================================
# TAB 2 — BATCH PREDICT
# ==================================================================
with tab_batch:
    st.markdown(
        '<div class="haveli-card">'
        '<h3>📂 Batch Prediction</h3>'
        '<p style="color:#C9B896;">Upload a CSV with columns: '
        '<b style="color:#E8B84B;">Age</b>, '
        '<b style="color:#E8B84B;">EstimatedSalary</b></p>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

        if not {"Age", "EstimatedSalary"}.issubset(batch_df.columns):
            st.error("CSV must contain 'Age' and 'EstimatedSalary' columns.")
        else:
            info  = available_models[selected_model_name]
            model = info["model"]

            X_batch = batch_df[["Age", "EstimatedSalary"]].copy()
            X_ready = scaler.transform(X_batch) if info["needs_scale"] else X_batch

            probs = model.predict_proba(X_ready)[:, 1]
            preds = (probs >= 0.5).astype(int)

            batch_df["Prediction"]  = np.where(preds == 1, "Buy", "No Buy")
            batch_df["Probability"] = np.round(probs, 4)
            batch_df["Persona"]     = [
                get_persona(int(a), int(s))
                for a, s in zip(batch_df["Age"], batch_df["EstimatedSalary"])
            ]

            st.success(f"✅ Predicted {len(batch_df)} customers using {selected_model_name}")
            st.dataframe(batch_df, use_container_width=True)

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

# ==================================================================
# TAB 3 — INSIGHTS
# ==================================================================
with tab_insights:
    st.markdown(
        '<div class="haveli-card"><h3>📊 Model Insights</h3></div>',
        unsafe_allow_html=True,
    )

    info  = available_models[selected_model_name]
    model = info["model"]

    # ---------------- Feature importance (trees) ----------------
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(
            model.feature_importances_,
            index=["Age", "EstimatedSalary"],
        ).sort_values()

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(
            fi.index, fi.values,
            color=["#E8B84B", "#D4A574"],
            edgecolor="#0F1419", linewidth=1.2,
        )
        ax.set_title(
            f"Feature Importance — {selected_model_name}",
            color="#E8B84B", fontweight="bold", fontsize=12,
        )
        ax.set_xlabel("Importance", color="#C9B896")
        ax.tick_params(colors="#F5EFE0")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#E8B84B")
            ax.spines[spine].set_alpha(0.4)
        fig.patch.set_facecolor("#1A2035")
        ax.set_facecolor("#1A2035")
        st.pyplot(fig)

    # ---------------- Coefficients (Logistic Regression) ----------------
    elif hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_[0], index=["Age", "EstimatedSalary"])
        st.markdown(
            '<p style="color:#C9A34E; font-weight:700; '
            'text-transform:uppercase; letter-spacing:0.6px;">'
            'Logistic Regression Coefficients (on scaled features):</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(coefs.rename("Coefficient").to_frame())

    # ---------------- KNN notice ----------------
    else:
        st.info(
            "KNN does not expose feature importances — it's an instance-based model. "
            "Try Random Forest or Decision Tree to see importances."
        )

    # ---------------- Training data snapshot ----------------
    data_path = "data/Social_Network_Ads.csv"
    if os.path.exists(data_path):
        st.markdown(
            '<div class="haveli-card"><h3>📁 Training Data Snapshot</h3></div>',
            unsafe_allow_html=True,
        )
        df_train = pd.read_csv(data_path)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '<p style="color:#C9A34E; font-weight:700; '
                'text-transform:uppercase; letter-spacing:0.6px;">'
                'Class distribution</p>',
                unsafe_allow_html=True,
            )
            st.bar_chart(df_train["Purchased"].value_counts(), color="#E8B84B")

        with c2:
            st.markdown(
                '<p style="color:#C9A34E; font-weight:700; '
                'text-transform:uppercase; letter-spacing:0.6px;">'
                'Age vs Salary (colored by Purchased)</p>',
                unsafe_allow_html=True,
            )
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(
                df_train["Age"], df_train["EstimatedSalary"],
                c=df_train["Purchased"], cmap="plasma",
                alpha=0.85, edgecolor="#0F1419", linewidth=0.4,
                s=42,
            )
            ax2.set_xlabel("Age", color="#C9B896")
            ax2.set_ylabel("Estimated Salary", color="#C9B896")
            ax2.set_title(
                "Purchase Distribution",
                color="#E8B84B", fontweight="bold", fontsize=12,
            )
            ax2.tick_params(colors="#F5EFE0")
            fig2.patch.set_facecolor("#1A2035")
            ax2.set_facecolor("#1A2035")
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax2.spines[spine].set_color("#E8B84B")
                ax2.spines[spine].set_alpha(0.4)
            st.pyplot(fig2)

# ==================================================================
# TAB 4 — HISTORY
# ==================================================================
with tab_history:
    st.markdown(
        '<div class="haveli-card"><h3>📜 Session Prediction History</h3></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download history",
            data=csv_hist,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the **Predict** tab and try one!")

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown(
    '<div class="haveli-footer">'
    '<b>Haveli ML Premium</b> — Built with Streamlit, Scikit-learn & a touch of heritage.'
    '</div>',
    unsafe_allow_html=True,
)