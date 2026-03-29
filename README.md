# Exam Score Analyzer & Feedback Generator

A Data Science project that analyzes student exam scores, clusters students into performance groups using K-Means, and generates personalized feedback reports.

---

## Project Structure

```
exam-score-analyzer/
├── data/                        # Auto-generated data & plots
├── model/                       # Saved K-Means model & scaler
├── Project_Report.pdf           # Full project report
├── README.md
├── analysis.py                  # Full pipeline (data → EDA → clustering → feedback)
└── app.py                       # Interactive student lookup app
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/exam-score-analyzer.git
cd exam-score-analyzer

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Run the analysis pipeline
python analysis.py

# 4. Launch the interactive app
python app.py
```

---

## How It Works

**analysis.py** runs the full pipeline in 4 steps:
1. Generates synthetic data for 200 students across 5 subjects
2. Performs EDA and saves 6 visualizations to `data/`
3. Applies K-Means clustering → saves model to `model/`
4. Generates personalized feedback → saves to `data/feedback_report.csv`

**app.py** lets you interactively:
- View class-wide group summaries
- Check subject averages
- Look up any student by ID
- List at-risk students

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas & numpy | Data handling |
| matplotlib & seaborn | Visualizations |
| scikit-learn | K-Means clustering |

---

## Author

**Your Name** | Roll No: XXXXXXXX | Your Institution
