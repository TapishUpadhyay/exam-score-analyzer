"""
app.py
=======
Interactive CLI app for the Exam Score Analyzer.
Lets you look up any student's feedback, view group stats,
and see subject-wise performance — no Streamlit required.

Usage:
    python app.py
"""

import os
import pandas as pd

FEEDBACK_PATH = "data/feedback_report.csv"
SUBJECTS      = ["Mathematics", "Science", "English", "History", "Computer_Science"]


def load_data():
    if not os.path.exists(FEEDBACK_PATH):
        print("\n[!] No feedback report found. Please run analysis.py first.")
        print("    Command: python analysis.py\n")
        exit(1)
    return pd.read_csv(FEEDBACK_PATH)


def print_header():
    print("\n" + "=" * 55)
    print("   Exam Score Analyzer — Interactive Lookup")
    print("=" * 55)


def show_group_summary(df):
    print("\n--- Class Overview ---")
    summary = df.groupby("Performance_Group").agg(
        Count=("Student_ID", "count"),
        Avg_Score=("Average", "mean"),
        Avg_Attendance=("Attendance_%", "mean")
    ).round(2)
    print(summary.to_string())
    print()


def show_subject_averages(df):
    print("\n--- Subject Averages ---")
    for subj in SUBJECTS:
        avg = df[subj].mean()
        bar = "#" * int(avg / 5)
        print(f"  {subj.replace('_',' '):<20} {avg:5.1f}  {bar}")
    print()


def lookup_student(df):
    sid = input("Enter Student ID (e.g. STU0042): ").strip().upper()
    row = df[df["Student_ID"] == sid]
    if row.empty:
        print(f"  [!] Student '{sid}' not found.")
    else:
        print("\n" + "─" * 50)
        print(row["Feedback"].values[0])
        print("─" * 50)


def show_at_risk(df):
    at_risk = df[df["Performance_Group"] == "At-Risk"][["Student_ID", "Average", "Attendance_%"]]
    print(f"\n--- At-Risk Students ({len(at_risk)} total) ---")
    print(at_risk.sort_values("Average").head(10).to_string(index=False))
    print()


def main():
    df = load_data()
    print_header()

    while True:
        print("\nOptions:")
        print("  1. Class overview (group summary)")
        print("  2. Subject averages")
        print("  3. Look up a student by ID")
        print("  4. Show at-risk students")
        print("  5. Exit")
        choice = input("\nEnter choice (1-5): ").strip()

        if   choice == "1": show_group_summary(df)
        elif choice == "2": show_subject_averages(df)
        elif choice == "3": lookup_student(df)
        elif choice == "4": show_at_risk(df)
        elif choice == "5":
            print("\nGoodbye!\n"); break
        else:
            print("  Invalid choice. Enter 1-5.")


if __name__ == "__main__":
    main()
