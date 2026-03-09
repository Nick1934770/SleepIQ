import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from supabase import create_client
import warnings

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SleepIQ",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .tip-card {
        background: #f0f4ff;
        border-left: 4px solid #4c6ef5;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
    }
    .auth-box {
        max-width: 420px;
        margin: 2rem auto;
        padding: 2rem;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model Training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on dataset…")
def load_and_train():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

    def label(x):
        if x >= 7:
            return "Good"
        elif x >= 5:
            return "Average"
        return "Poor"

    df["SleepQuality"] = df["Quality of Sleep"].apply(label)

    features = [
        "Sleep Duration",
        "Stress Level",
        "Physical Activity Level",
        "Daily Steps",
        "Heart Rate",
        "BMI Category",
    ]
    X = df[features]
    y = df["SleepQuality"]

    numeric = ["Sleep Duration", "Stress Level", "Physical Activity Level", "Daily Steps", "Heart Rate"]
    categorical = ["BMI Category"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))

    return pipeline, accuracy, df


pipeline, model_accuracy, df = load_and_train()
CLASS_ORDER = ["Poor", "Average", "Good"]
BMI_OPTIONS = sorted(df["BMI Category"].dropna().unique().tolist())

# ── Supabase Client (cached) ──────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# ── Session State ─────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "username" not in st.session_state:
    st.session_state.username = None

# ── Handle Email Confirmation Callback ───────────────────────────────────────
# When the user clicks the confirmation link in their email, Supabase redirects
# them back to the app with ?code=xxx in the URL. We exchange that code for a
# live session and log them in automatically.
params = st.query_params
if "code" in params and st.session_state.user is None:
    try:
        result = get_supabase().auth.exchange_code_for_session(params["code"])
        st.session_state.user = result.user
        profile = (
            get_supabase()
            .table("profiles")
            .select("username")
            .eq("id", result.user.id)
            .single()
            .execute()
        )
        st.session_state.username = profile.data["username"]
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Email confirmation failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN — shown when not logged in
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.user is None:
    st.markdown('<h1 class="main-header">🌙 SleepIQ</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered sleep quality prediction — create an account to get started</p>',
        unsafe_allow_html=True,
    )

    _, col_auth, _ = st.columns([1, 1.2, 1])

    with col_auth:
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        # ── Log In ─────────────────────────────────────────────────────────────
        with tab_login:
            st.write("")
            login_email    = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            st.write("")

            if st.button("Log In", use_container_width=True, type="primary", key="login_btn"):
                if not login_email or not login_password:
                    st.error("Please enter your email and password.")
                else:
                    try:
                        result = get_supabase().auth.sign_in_with_password({
                            "email": login_email,
                            "password": login_password,
                        })
                        st.session_state.user = result.user
                        profile = (
                            get_supabase()
                            .table("profiles")
                            .select("username")
                            .eq("id", result.user.id)
                            .single()
                            .execute()
                        )
                        st.session_state.username = profile.data["username"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")

        # ── Sign Up ────────────────────────────────────────────────────────────
        with tab_signup:
            st.write("")
            signup_username = st.text_input("Username", key="signup_username")
            signup_email    = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password",
                                            help="At least 6 characters")
            st.write("")

            if st.button("Create Account", use_container_width=True, type="primary", key="signup_btn"):
                if not signup_username or not signup_email or not signup_password:
                    st.error("Please fill in all fields.")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        # Create the auth user — Supabase sends a confirmation email
                        result = get_supabase().auth.sign_up({
                            "email": signup_email,
                            "password": signup_password,
                            "options": {
                                "email_redirect_to": st.secrets.get(
                                    "REDIRECT_URL", "http://localhost:8501"
                                )
                            },
                        })
                        # Save username to profiles table
                        get_supabase().table("profiles").insert({
                            "id": result.user.id,
                            "username": signup_username,
                        }).execute()

                        st.success(
                            f"Account created! Check **{signup_email}** for a confirmation link. "
                            "Click it and you'll be logged in automatically."
                        )
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")

    st.stop()  # Don't render any of the main app below

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — shown when logged in
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    st.caption(st.session_state.user.email)
    st.divider()
    if st.button("Log Out", use_container_width=True):
        get_supabase().auth.sign_out()
        st.session_state.user = None
        st.session_state.username = None
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🌙 SleepIQ</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Enter your daily habits — get an AI-powered sleep quality prediction with personalised tips</p>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_about = st.tabs(["🔮 Predict", "📊 Dataset Insights", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Your Daily Metrics")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        sleep_duration = st.slider(
            "🛏️ Sleep Duration (hours/night)",
            min_value=4.0, max_value=12.0, value=7.0, step=0.5,
            help="How many hours do you typically sleep per night?",
        )
        stress_level = st.slider(
            "😰 Stress Level  (1 = Very Low · 10 = Very High)",
            min_value=1, max_value=10, value=5,
            help="Rate your overall daily stress level.",
        )
        physical_activity = st.slider(
            "🏃 Physical Activity (min/day)",
            min_value=0, max_value=120, value=45, step=5,
            help="Average minutes of moderate-to-vigorous exercise per day.",
        )

    with col_right:
        daily_steps = st.number_input(
            "👟 Daily Steps",
            min_value=0, max_value=30000, value=8000, step=500,
            help="Average steps per day (from a fitness tracker or estimate).",
        )
        heart_rate = st.number_input(
            "❤️ Resting Heart Rate (bpm)",
            min_value=40, max_value=120, value=70,
            help="Your resting heart rate in beats per minute.",
        )
        bmi_category = st.selectbox(
            "⚖️ BMI Category",
            options=BMI_OPTIONS,
            help="Your Body Mass Index category.",
        )

    st.write("")
    predict_clicked = st.button("🔮 Predict My Sleep Quality", use_container_width=True, type="primary")

    if predict_clicked:
        input_df = pd.DataFrame({
            "Sleep Duration": [sleep_duration],
            "Stress Level": [stress_level],
            "Physical Activity Level": [physical_activity],
            "Daily Steps": [daily_steps],
            "Heart Rate": [heart_rate],
            "BMI Category": [bmi_category],
        })

        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        prob_map = dict(zip(pipeline.classes_, probabilities))

        # ── Save to Supabase ───────────────────────────────────────────────────
        try:
            get_supabase().table("sleep_predictions").insert({
                "user_id": st.session_state.user.id,
                "sleep_duration": sleep_duration,
                "stress_level": stress_level,
                "physical_activity": physical_activity,
                "daily_steps": int(daily_steps),
                "heart_rate": int(heart_rate),
                "bmi_category": bmi_category,
                "prediction": prediction,
                "good_probability": round(float(prob_map.get("Good", 0)), 4),
            }).execute()
        except Exception as e:
            st.warning(f"Could not save prediction to database: {e}")

        st.divider()

        # ── Result Banner ──────────────────────────────────────────────────────
        COLOR_MAP = {"Good": "#28a745", "Average": "#e67e00", "Poor": "#dc3545"}
        EMOJI_MAP = {"Good": "😴", "Average": "😐", "Poor": "😟"}
        LABEL_MAP = {
            "Good": "Great sleep quality! Keep it up.",
            "Average": "Decent — but room to improve.",
            "Poor": "Your habits need some attention.",
        }

        col_badge, col_gauge = st.columns([1, 2], gap="large")

        with col_badge:
            color = COLOR_MAP[prediction]
            emoji = EMOJI_MAP[prediction]
            label_text = LABEL_MAP[prediction]
            st.markdown(
                f"""
                <div style="background:{color}18; border:2px solid {color};
                            border-radius:12px; padding:1.5rem; text-align:center;">
                    <div style="font-size:3.5rem">{emoji}</div>
                    <div style="font-size:1.6rem; font-weight:700; color:{color}">
                        {prediction} Sleep
                    </div>
                    <div style="color:#555; margin-top:0.3rem; font-size:0.9rem">
                        {label_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_gauge:
            good_pct = prob_map.get("Good", 0) * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(good_pct, 1),
                number={"suffix": "%"},
                title={"text": "Probability of Good Sleep"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#4c6ef5"},
                    "steps": [
                        {"range": [0, 33], "color": "#ffe0e0"},
                        {"range": [33, 66], "color": "#fff4cc"},
                        {"range": [66, 100], "color": "#d4f5d4"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Confidence Bars ────────────────────────────────────────────────────
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({
            "Category": CLASS_ORDER,
            "Probability (%)": [prob_map.get(c, 0) * 100 for c in CLASS_ORDER],
        })
        fig_bar = px.bar(
            prob_df,
            x="Category",
            y="Probability (%)",
            color="Category",
            color_discrete_map={"Good": "#28a745", "Average": "#e67e00", "Poor": "#dc3545"},
            text_auto=".1f",
        )
        fig_bar.update_layout(
            showlegend=False,
            height=280,
            margin=dict(t=10, b=10),
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Habit Health Scores ────────────────────────────────────────────────
        st.subheader("Your Habits at a Glance")

        sleep_score    = min(sleep_duration / 8 * 10, 10)
        stress_score   = max((10 - stress_level), 0)
        activity_score = min(physical_activity / 60 * 10, 10)
        steps_score    = min(daily_steps / 10000 * 10, 10)
        hr_score       = max(10 - (heart_rate - 60) / 6, 0)
        bmi_score      = {"Normal": 10, "Normal Weight": 10, "Overweight": 6, "Obese": 3, "Underweight": 5}.get(bmi_category, 7)

        categories = ["Sleep Duration", "Stress Level", "Physical Activity", "Daily Steps", "Heart Rate", "BMI"]
        values     = [sleep_score, stress_score, activity_score, steps_score, hr_score, bmi_score]
        BAR_COLORS = ["#4c6ef5", "#f03e3e", "#37b24d", "#f59f00", "#e64980", "#7950f2"]

        habits_df = pd.DataFrame({"Habit": categories, "Score": values})
        fig_habits = px.bar(
            habits_df, x="Habit", y="Score", color="Habit",
            color_discrete_sequence=BAR_COLORS, text="Score",
            title="Your Habit Health Scores  (out of 10)",
        )
        fig_habits.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_habits.update_layout(
            showlegend=False, height=380,
            yaxis=dict(range=[0, 11], title="Score / 10"),
            xaxis_title="", margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_habits, use_container_width=True)

        # ── Personalised Tips ──────────────────────────────────────────────────
        st.subheader("💡 Personalised Recommendations")

        tips = []
        if sleep_duration < 7:
            tips.append(("🛏️ Sleep More", "Aim for **7–9 hours** per night. Going to bed 30 minutes earlier is a great first step."))
        elif sleep_duration > 9:
            tips.append(("🛏️ Watch Oversleeping", "Consistently sleeping >9 hrs can signal an underlying issue. Try a consistent wake time."))
        if stress_level >= 7:
            tips.append(("🧘 Reduce Stress", "High stress disrupts deep sleep. Try **5-min box breathing** or journaling before bed."))
        elif stress_level >= 5:
            tips.append(("😌 Manage Mild Stress", "Your stress is moderate. A short walk in the evening can lower cortisol levels."))
        if physical_activity < 30:
            tips.append(("🏃 Move More", "Even **30 minutes of brisk walking** daily can meaningfully improve sleep quality."))
        if daily_steps < 5000:
            tips.append(("👟 Increase Steps", "Aim for **8,000–10,000 steps/day**. Take the stairs or walk during lunch breaks."))
        if heart_rate > 80:
            tips.append(("❤️ Lower Resting HR", "A resting HR >80 bpm can reduce sleep depth. Aerobic exercise and relaxation techniques help."))
        if bmi_category in ["Overweight", "Obese"]:
            tips.append(("⚖️ Work on BMI", "Excess weight increases risk of sleep apnea. Even a 5–10% weight reduction can improve sleep."))
        if prediction == "Good" and not tips:
            tips.append(("🌟 You're Doing Great!", "Your metrics are well-balanced. Maintain consistency — regular schedules are key to great sleep."))
        if not tips:
            tips.append(("✅ Solid Habits", "Your inputs look healthy. Keep maintaining your current lifestyle."))

        for title, desc in tips:
            st.markdown(f'<div class="tip-card"><b>{title}</b> — {desc}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("Dataset Overview")

    def label(x):
        return "Good" if x >= 7 else ("Average" if x >= 5 else "Poor")

    df["SleepQuality"] = df["Quality of Sleep"].apply(label)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", len(df))
    k2.metric("Good Sleepers", f"{(df['SleepQuality'] == 'Good').mean()*100:.1f}%")
    k3.metric("Avg Sleep", f"{df['Sleep Duration'].mean():.1f} hrs")
    k4.metric("Model Accuracy", f"{model_accuracy*100:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df, x="Quality of Sleep", nbins=10,
            color_discrete_sequence=["#4c6ef5"],
            title="Distribution of Sleep Quality Scores",
            labels={"Quality of Sleep": "Sleep Quality (1–10)"},
        )
        fig_hist.update_layout(height=320)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(
            df, x="BMI Category", y="Quality of Sleep",
            color="BMI Category",
            title="Sleep Quality by BMI Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_box.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    fig_scatter = px.scatter(
        df, x="Stress Level", y="Quality of Sleep",
        color="SleepQuality",
        color_discrete_map={"Good": "#28a745", "Average": "#e67e00", "Poor": "#dc3545"},
        size="Physical Activity Level", opacity=0.7,
        title="Stress Level vs Sleep Quality  (bubble size = Physical Activity)",
        labels={"Quality of Sleep": "Sleep Quality (1–10)"},
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    numeric_cols = ["Sleep Duration", "Quality of Sleep", "Stress Level",
                    "Physical Activity Level", "Daily Steps", "Heart Rate"]
    corr = df[numeric_cols].corr().round(2)
    fig_heat = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=True, title="Feature Correlation Matrix", aspect="auto",
    )
    fig_heat.update_layout(height=420)
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("About This App")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
### How It Works
This app uses a **Random Forest Classifier** trained on the
*Sleep Health & Lifestyle Dataset* to classify your sleep quality into three tiers:

| Category | Score |
|---|---|
| 🟢 Good | 7 – 10 |
| 🟡 Average | 5 – 6 |
| 🔴 Poor | 1 – 4 |

The model uses a **scikit-learn Pipeline** with:
- `ColumnTransformer` for StandardScaler (numeric) + OneHotEncoder (BMI)
- `RandomForestClassifier` (300 trees, depth 12)
- `@st.cache_resource` so the model trains once and is reused
        """)

    with col_b:
        st.markdown("""
### Input Features

| Feature | Description |
|---|---|
| Sleep Duration | Hours of sleep per night |
| Stress Level | Daily stress score (1–10) |
| Physical Activity | Exercise in minutes/day |
| Daily Steps | Average steps per day |
| Heart Rate | Resting HR in bpm |
| BMI Category | Body Mass Index class |

### Dataset
The **Sleep Health & Lifestyle Dataset** contains 374 individuals
and 13 health/lifestyle variables collected to study sleep patterns.
        """)
