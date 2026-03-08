# 🌙 SleepIQ — AI-Powered Sleep Quality Predictor

SleepIQ is a machine learning web app that predicts your sleep quality based on your daily health habits and gives you personalised recommendations to improve it.

Built with **Streamlit**, **scikit-learn**, and **Plotly**.

---

## 📸 Features

- 🔮 **Sleep Quality Prediction** — enter your daily metrics and get an instant Good / Average / Poor prediction
- 📊 **Confidence Gauge** — see the probability breakdown across all three categories
- 🏅 **Habit Health Scores** — your 6 key habits scored out of 10 and visualised as a bar chart
- 💡 **Personalised Tips** — targeted recommendations based on your specific inputs
- 📈 **Dataset Insights** — explore the underlying data through interactive charts including a correlation heatmap, box plots, and scatter plots
- ⚡ **Cached Model** — the Random Forest model trains once and is reused across sessions for fast load times

---

## 🧠 How It Works

The app uses a **Random Forest Classifier** trained on the [Sleep Health & Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) (374 individuals, 13 variables).

### Input Features
| Feature | Description |
|---|---|
| Sleep Duration | Hours of sleep per night |
| Stress Level | Daily stress score (1–10) |
| Physical Activity | Exercise in minutes per day |
| Daily Steps | Average steps per day |
| Heart Rate | Resting heart rate in bpm |
| BMI Category | Body Mass Index classification |

### Sleep Quality Labels
| Label | Quality of Sleep Score |
|---|---|
| 🟢 Good | 7 – 10 |
| 🟡 Average | 5 – 6 |
| 🔴 Poor | 1 – 4 |

### ML Pipeline
- `ColumnTransformer` — `StandardScaler` for numeric features, `OneHotEncoder` for BMI Category
- `RandomForestClassifier` — 300 trees, max depth 12
- `@st.cache_resource` — model is trained once and cached for the app's lifetime
- **Test accuracy: ~98.7%**

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sleepiq.git
cd sleepiq

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run Sleep_App.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
sleepiq/
├── Sleep_App.py                            # Main Streamlit application
├── Sleep_health_and_lifestyle_dataset.csv  # Training dataset
├── data.py                                 # Exploratory data analysis script
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore rules
└── README.md                              # This file
```

---

## 🌐 Deployment

This app is designed to be deployed on **Streamlit Community Cloud** (free).

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and set the main file to `Sleep_App.py`
5. Click **Deploy**

HTTPS and a public URL are provisioned automatically.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | Machine learning pipeline and model |
| `plotly` | Interactive charts and visualisations |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
