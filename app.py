
# app.py
# Streamlit AI Personal Finance Prototype (4-hour build)
# Author: M365 Copilot

import os
import io
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Personal Finance Tracker", page_icon="💸", layout="wide")

# -----------------------------
# Utility & Seed Data
# -----------------------------
CATEGORIES = [
    "Income",
    "Rent & Housing",
    "Food & Dining",
    "Groceries",
    "Transport",
    "Fuel",
    "Shopping",
    "Subscriptions",
    "Utilities & Bills",
    "Healthcare",
    "Education",
    "Insurance",
    "Entertainment",
    "Fees & Charges",
    "Transfers",
    "Miscellaneous"
]

KEYWORD_RULES = {
    "Income": ["salary", "payroll", "stipend", "credit interest"],
    "Rent & Housing": ["rent", "landlord", "lease", "housing"],
    "Food & Dining": ["restaurant", "cafe", "coffee", "starbucks", "eating", "dining"],
    "Groceries": ["grocery", "groceries", "big bazaar", "dmart", "more", "supermarket"],
    "Transport": ["uber", "ola", "metro", "bus", "train", "flight", "ride"],
    "Fuel": ["fuel", "petrol", "diesel", "bpcl", "hpcl", "iocl"],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "retail"],
    "Subscriptions": ["netflix", "prime", "spotify", "subscription", "subscr"],
    "Utilities & Bills": ["electricity", "water", "gas", "utilities", "bill", "broadband", "internet"],
    "Healthcare": ["pharmacy", "medical", "doctor", "clinic", "hospital"],
    "Education": ["tuition", "school", "college", "course", "training"],
    "Insurance": ["insurance", "premium", "policy"],
    "Entertainment": ["movie", "cinema", "concert", "event", "entertainment"],
    "Fees & Charges": ["fee", "charge", "atm fee", "late fee"],
    "Transfers": ["transfer", "imps", "neft", "rtgs", "upi", "to self"],
    "Miscellaneous": []
}

SEED_LABELED = pd.DataFrame({
    'description': [
        'Starbucks coffee', 'Rent payment to landlord', 'Uber ride BLR', 'DMart grocery shopping',
        'Monthly salary credit', 'Electricity bill BESCOM', 'Petrol bunk fill HPCL',
        'Amazon order electronics', 'Health pharmacy purchase', 'Tuition fee',
        'Netflix subscription', 'Movie tickets PVR', 'Bank fee charge', 'Phone recharge Jio',
        'Transfer to savings', 'Insurance premium LIC'
    ],
    'category': [
        'Food & Dining', 'Rent & Housing', 'Transport', 'Groceries', 'Income', 'Utilities & Bills',
        'Fuel', 'Shopping', 'Healthcare', 'Education', 'Subscriptions', 'Entertainment',
        'Fees & Charges', 'Utilities & Bills', 'Transfers', 'Insurance'
    ]
})

# -----------------------------
# ML Model (TF-IDF + LogisticRegression)
# -----------------------------
@st.cache_resource
def build_seed_model():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=300))
    ])
    pipe.fit(SEED_LABELED['description'], SEED_LABELED['category'])
    return pipe

MODEL = build_seed_model()

# -----------------------------
# Categorization Logic (Hybrid)
# -----------------------------

def rule_based_category(text: str) -> str:
    if not text:
        return "Miscellaneous"
    t = text.lower()
    for cat, kws in KEYWORD_RULES.items():
        for kw in kws:
            if kw in t:
                return cat
    # Heuristic based on sign words
    if any(w in t for w in ["salary", "credit"]):
        return "Income"
    return "Miscellaneous"


def ml_category(text: str) -> str:
    try:
        return MODEL.predict([text])[0]
    except Exception:
        return "Miscellaneous"


def hybrid_categorize(desc: str) -> str:
    rb = rule_based_category(desc)
    if rb != "Miscellaneous":
        return rb
    return ml_category(desc)

# -----------------------------
# Helper Functions
# -----------------------------

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: date, description, amount (positive for spend, negative for income OR vice versa)
    cols = {c.lower(): c for c in df.columns}
    # Standardize column names
    def getcol(name):
        for c in df.columns:
            if c.lower() == name:
                return c
        return None
    date_col = getcol('date')
    desc_col = getcol('description') or getcol('details') or getcol('narration')
    amount_col = getcol('amount')

    if not (date_col and desc_col and amount_col):
        raise ValueError("CSV must include columns: date, description, amount")

    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'description': df[desc_col].astype(str),
        'amount': pd.to_numeric(df[amount_col], errors='coerce')
    })
    out = out.dropna(subset=['date', 'description', 'amount']).copy()
    out['month'] = out['date'].dt.to_period('M').astype(str)
    # Direction: debit(negative cash flow) vs credit
    out['direction'] = np.where(out['amount'] < 0, 'credit', 'debit')
    return out


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['category'] = df['description'].apply(hybrid_categorize)
    # If amount < 0 and category not Income, consider Transfers/Income
    df.loc[(df['amount'] < 0) & (df['category'] == 'Miscellaneous'), 'category'] = 'Income'
    return df


def compute_insights(df: pd.DataFrame) -> dict:
    insights = {}
    # Category totals (only debits/spend)
    spend_df = df[df['amount'] > 0]
    cat_totals = spend_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    insights['top_categories'] = cat_totals.head(5)

    # Monthly trends
    monthly = spend_df.groupby(['month', 'category'])['amount'].sum().reset_index()
    insights['monthly'] = monthly

    # Spike detection: change vs previous month per category
    spikes = []
    for cat in monthly['category'].unique():
        m = monthly[monthly['category'] == cat].copy()
        m['month_dt'] = pd.PeriodIndex(m['month'], freq='M').to_timestamp()
        m = m.sort_values('month_dt')
        m['prev'] = m['amount'].shift(1)
        m['pct_change'] = (m['amount'] - m['prev']) / m['prev']
        for _, row in m.dropna().iterrows():
            if row['pct_change'] >= 0.5 and row['amount'] >= 1000:  # configurable threshold
                spikes.append({
                    'category': cat,
                    'month': row['month'],
                    'change': f"+{int(row['pct_change']*100)}%",
                    'amount': row['amount']
                })
    insights['spikes'] = spikes
    return insights


def simple_qa(df: pd.DataFrame, q: str) -> str:
    """Very basic Q&A without external LLM.
    Supports questions like:
    - How much did I spend on <category> in <month>?
    - What is my total spending in <month>?
    - Top categories this month
    """
    ql = q.lower()
    # Extract month token like 'nov', 'november', '2025-11'
    month = None
    months_map = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    for m, num in months_map.items():
        if m in ql:
            # Assume current year if not provided
            year = datetime.now().year
            month = f"{year}-{num}"
            break
    # ISO yyyy-mm in text
    m = re.search(r"(20\d{2})-(0[1-9]|1[0-2])", ql)
    if m:
        month = m.group(0)

    # Extract category by matching known categories
    cat = None
    for c in CATEGORIES:
        if c.lower() in ql:
            cat = c
            break
    # Heuristic mapping
    if not cat:
        hints = {
            'food': 'Food & Dining', 'dining': 'Food & Dining', 'coffee': 'Food & Dining',
            'grocer': 'Groceries', 'rent': 'Rent & Housing', 'transport': 'Transport',
            'fuel': 'Fuel', 'amazon': 'Shopping', 'shop': 'Shopping', 'movie': 'Entertainment',
            'electricity': 'Utilities & Bills', 'bill': 'Utilities & Bills', 'insurance': 'Insurance'
        }
        for k, v in hints.items():
            if k in ql:
                cat = v
                break

    df2 = df.copy()
    if month:
        df2 = df2[df2['month'] == month]
    spend_df = df2[df2['amount'] > 0]

    if 'total' in ql and 'spend' in ql:
        total = spend_df['amount'].sum()
        return f"Total spending{f' in {month}' if month else ''}: ₹{total:,.0f}"

    if cat:
        total = spend_df[spend_df['category'] == cat]['amount'].sum()
        return f"Spending{f' in {month}' if month else ''} on {cat}: ₹{total:,.0f}"

    # Top categories
    if 'top' in ql and 'category' in ql:
        tops = spend_df.groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
        return "Top categories" + (f" in {month}" if month else "") + ": " + \
               ", ".join([f"{i} (₹{v:,.0f})" for i, v in tops.items()])

    return "I can answer: total spending, spending by category/month, and top categories. Try: 'How much did I spend on Food in 2025-11?'"

# -----------------------------
# UI
# -----------------------------

st.title("💸 AI Personal Finance Tracker (Prototype)")
st.caption("Upload a CSV of transactions (date, description, amount). The app will auto-categorize and show insights.")

with st.sidebar:
    st.header("Quick Start")
    sample_df = pd.DataFrame({
        'date': pd.date_range('2025-10-01', periods=24, freq='3D'),
        'description': [
            'Starbucks coffee', 'Rent payment to landlord', 'Uber ride', 'DMart grocery shopping',
            'Monthly salary credit', 'Electricity bill', 'HPCL petrol bunk', 'Amazon order',
            'Pharmacy purchase', 'Tuition fee', 'Netflix subscription', 'Movie tickets',
            'Bank fee', 'Phone recharge', 'Transfer to savings', 'Insurance premium',
            'Cafe dinner', 'Metro tickets', 'Diesel refill', 'Online shopping',
            'Doctor visit', 'School supplies', 'Spotify subscription', 'Concert passes'
        ],
        'amount': [275, 25000, 320, 1800, -120000, 2300, 4000, 3500, 950, 20000, 649, 1200, 50, 299, 50000, 15000,
                   800, 300, 4500, 2200, 1800, 1200, 149, 4500]
    })
    buf = io.StringIO()
    sample_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download sample CSV", buf.getvalue(), file_name="sample_transactions.csv", mime="text/csv")

uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        df = normalize_dataframe(raw)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.subheader("Raw Transactions (Normalized)")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Auto-categorization")
    model_choice = st.radio("Categorization Mode", ["Hybrid (Rules + ML)", "Rules only", "ML only"], index=0)

    df_cat = df.copy()
    if model_choice == "Rules only":
        df_cat['category'] = df_cat['description'].apply(rule_based_category)
    elif model_choice == "ML only":
        df_cat['category'] = df_cat['description'].apply(ml_category)
    else:
        df_cat['category'] = df_cat['description'].apply(hybrid_categorize)

    st.dataframe(df_cat.head(100), use_container_width=True)

    st.subheader("Edit & Correct Categories")
    df_edit = st.data_editor(df_cat, num_rows="dynamic", use_container_width=True, key="editor")

    # Insights
    st.subheader("Insights")
    insights = compute_insights(df_edit)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Categories (Spend)**")
        if not insights['top_categories'].empty:
            st.bar_chart(insights['top_categories'])
        else:
            st.info("No spend rows detected.")
    with col2:
        st.markdown("**Monthly Trends by Category**")
        if not insights['monthly'].empty:
            pivot = insights['monthly'].pivot(index='month', columns='category', values='amount').fillna(0)
            st.line_chart(pivot)
        else:
            st.info("No spend rows detected.")

    if insights['spikes']:
        with st.expander("⚠️ Spend Spikes Detected"):
            st.table(pd.DataFrame(insights['spikes']))

    # Simple Q&A
    st.subheader("Ask a question")
    q = st.text_input("e.g., 'How much did I spend on Food in 2025-11?' or 'Total spending in Nov' ")
    if q:
        answer = simple_qa(df_edit, q)
        st.success(answer)

    # Export
    st.subheader("Export Categorized CSV")
    out_buf = io.StringIO()
    df_edit.to_csv(out_buf, index=False)
    st.download_button("⬇️ Download categorized CSV", out_buf.getvalue(), file_name="categorized_transactions.csv", mime="text/csv")
else:
    st.info("Upload a CSV to begin. You can download the sample from the sidebar.")

# Footer
st.caption("Local-only prototype. No data leaves your machine.")
