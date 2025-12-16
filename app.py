
# app.py
# Streamlit AI Personal Finance Tracker – with Budgeting & Investment Planner
# Author: M365 Copilot
# Disclaimer: Educational demo only. Not financial advice. Consult a licensed advisor.

import os
import io
import re
import json
import calendar
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Finovo- Ai Enabled Personal Finance Tracker", page_icon="💸", layout="wide")

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
    # Expect columns: date, description, amount
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
    out['direction'] = np.where(out['amount'] < 0, 'credit', 'debit')
    return out


def compute_insights(df: pd.DataFrame) -> dict:
    insights = {}
    spend_df = df[df['amount'] > 0]
    cat_totals = spend_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    insights['top_categories'] = cat_totals.head(5)
    monthly = spend_df.groupby(['month', 'category'])['amount'].sum().reset_index()
    insights['monthly'] = monthly
    # spikes
    spikes = []
    for cat in monthly['category'].unique():
        m = monthly[monthly['category'] == cat].copy()
        m['month_dt'] = pd.PeriodIndex(m['month'], freq='M').to_timestamp()
        m = m.sort_values('month_dt')
        m['prev'] = m['amount'].shift(1)
        m['pct_change'] = (m['amount'] - m['prev']) / m['prev']
        for _, row in m.dropna().iterrows():
            if row['pct_change'] >= 0.5 and row['amount'] >= 1000:
                spikes.append({
                    'category': cat,
                    'month': row['month'],
                    'change': f"+{int(row['pct_change']*100)}%",
                    'amount': row['amount']
                })
    insights['spikes'] = spikes
    return insights

# -----------------------------
# Budgeting Helpers
# -----------------------------

def suggest_budgets(df: pd.DataFrame, months_back: int = 3) -> dict:
    """Suggest budgets based on average spend over the last N months per category."""
    spend_df = df[df['amount'] > 0].copy()
    # Choose last N distinct months
    months = sorted(spend_df['month'].unique())
    months = months[-months_back:] if len(months) >= months_back else months
    avg = spend_df[spend_df['month'].isin(months)].groupby('category')['amount'].mean()
    budgets = {cat: float(avg.get(cat, 0)) for cat in CATEGORIES if cat != 'Income'}
    return budgets


def get_month_bounds(month_str: str):
    year, mon = map(int, month_str.split('-'))
    first = date(year, mon, 1)
    days_in_month = calendar.monthrange(year, mon)[1]
    last = date(year, mon, days_in_month)
    return first, last, days_in_month


def budget_progress(df: pd.DataFrame, month: str, budgets: dict):
    spend_month = df[(df['month'] == month) & (df['amount'] > 0)]
    used = spend_month.groupby('category')['amount'].sum()
    first, last, days_in_month = get_month_bounds(month)
    today = datetime.now().date()
    days_elapsed = (min(today, last) - first).days + 1 if today >= first else 0
    run_rate_factor = days_in_month / max(days_elapsed, 1)

    rows = []
    for cat, limit in budgets.items():
        used_amt = float(used.get(cat, 0.0))
        pct = (used_amt / limit) if limit > 0 else 0.0
        # forecast end-of-month
        forecast = used_amt * run_rate_factor if days_elapsed > 0 else used_amt
        overshoot = (forecast > limit) if limit > 0 else False
        rows.append({
            'category': cat,
            'limit': limit,
            'used': used_amt,
            'pct_used': pct,
            'forecast_eom': forecast,
            'overshoot': overshoot
        })
    return pd.DataFrame(rows)

# -----------------------------
# Investment Planner (Educational)
# -----------------------------

def monthly_income_expense(df: pd.DataFrame, month: str):
    month_df = df[df['month'] == month]
    income = -month_df[month_df['amount'] < 0]['amount'].sum()  # negatives as income
    expenses = month_df[month_df['amount'] > 0]['amount'].sum()
    return float(income), float(expenses)


def recommended_allocation(surplus: float, profile: str):
    """Return allocation percentages by profile (no product recommendations)."""
    profiles = {
        'Conservative': {'Debt/FI': 0.70, 'Equity Index': 0.20, 'Gold': 0.10},
        'Balanced': {'Debt/FI': 0.40, 'Equity Index': 0.50, 'Gold': 0.10},
        'Growth': {'Debt/FI': 0.20, 'Equity Index': 0.70, 'Gold': 0.10},
    }
    alloc = profiles.get(profile, profiles['Balanced'])
    return {k: round(surplus * v, 2) for k, v in alloc.items()}


def investment_plan(df: pd.DataFrame, month: str, emergency_months: int, emergency_current: float,
                    debt_amount: float, debt_rate: float, profile: str,
                    emergency_cap_pct: float = 0.3, debt_cap_pct: float = 0.3):
    """Compute a simple educational plan based on surplus and priorities."""
    income, expenses = monthly_income_expense(df, month)
    surplus = income - expenses
    plan = {
        'income': income,
        'expenses': expenses,
        'surplus': surplus,
        'notes': []
    }

    if surplus <= 0:
        plan['notes'].append("Deficit detected. Consider reducing discretionary spending or adjusting budgets.")
        plan['allocations'] = {}
        return plan

    # Emergency fund target
    target_emergency = expenses * emergency_months
    emergency_gap = max(0.0, target_emergency - emergency_current)
    emergency_alloc = min(surplus * emergency_cap_pct, emergency_gap)

    # Debt prepayment (if rate is high)
    debt_alloc = 0.0
    if debt_amount > 0 and debt_rate >= 0.12:
        debt_alloc = min(surplus * debt_cap_pct, debt_amount)
        plan['notes'].append("High-interest debt detected (≥12%). Prioritising prepayment up to cap.")

    remaining = surplus - emergency_alloc - debt_alloc
    market_alloc = recommended_allocation(max(0.0, remaining), profile)

    plan['allocations'] = {
        'Emergency Fund (cash/FDs)': round(emergency_alloc, 2),
        'Debt Prepayment': round(debt_alloc, 2),
        **market_alloc
    }

    # Explanations
    if emergency_gap > 0:
        plan['notes'].append(
            f"Emergency fund gap: ₹{emergency_gap:,.0f} (target {emergency_months} months × expenses)."
        )
    if remaining <= 0:
        plan['notes'].append("Surplus fully allocated to safety/debt priorities this month.")
    else:
        plan['notes'].append(f"Remaining surplus allocated per '{profile}' risk profile.")

    return plan

# -----------------------------
# Simple Q&A
# -----------------------------

def simple_qa(df: pd.DataFrame, q: str) -> str:
    ql = q.lower()
    month = None
    months_map = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    for m, num in months_map.items():
        if m in ql:
            year = datetime.now().year
            month = f"{year}-{num}"
            break
    m = re.search(r"(20\d{2})-(0[1-9]|1[0-2])", ql)
    if m:
        month = m.group(0)

    cat = None
    for c in CATEGORIES:
        if c.lower() in ql:
            cat = c
            break
    hints = {
        'food': 'Food & Dining', 'dining': 'Food & Dining', 'coffee': 'Food & Dining',
        'grocer': 'Groceries', 'rent': 'Rent & Housing', 'transport': 'Transport',
        'fuel': 'Fuel', 'amazon': 'Shopping', 'shop': 'Shopping', 'movie': 'Entertainment',
        'electricity': 'Utilities & Bills', 'bill': 'Utilities & Bills', 'insurance': 'Insurance'
    }
    if not cat:
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

    if 'top' in ql and 'category' in ql:
        tops = spend_df.groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
        return "Top categories" + (f" in {month}" if month else "") + ": " + \
               ", ".join([f"{i} (₹{v:,.0f})" for i, v in tops.items()])

    return "I can answer: total spending, spending by category/month, and top categories. Try: 'How much did I spend on Food in 2025-11?'"

# -----------------------------
# UI
# -----------------------------

st.title("💸 AI Personal Finance Tracker (Prototype)")
st.caption("Educational demo. Not financial advice. Your data stays local.")

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

    # Auto-categorization
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

    # -----------------------------
    # Budgeting UI
    # -----------------------------
    st.subheader("Budgets (Monthly)")
    months_available = sorted(df_edit['month'].unique())
    default_month = months_available[-1] if months_available else None
    sel_month = st.selectbox("Select month", months_available, index=len(months_available)-1 if months_available else 0)

    if 'budgets' not in st.session_state:
        st.session_state['budgets'] = {cat: 0.0 for cat in CATEGORIES if cat != 'Income'}

    colA, colB = st.columns([2,1])
    with colA:
        st.write("Enter budget limits (₹) per category for ", sel_month)
        for cat in [c for c in CATEGORIES if c != 'Income']:
            st.session_state['budgets'][cat] = st.number_input(f"{cat}", min_value=0.0, step=100.0, value=float(st.session_state['budgets'].get(cat, 0.0)))
    with colB:
        if st.button("Auto-suggest from recent months"):
            suggested = suggest_budgets(df_edit)
            st.session_state['budgets'].update(suggested)
            st.success("Budgets updated from recent averages.")

    bp = budget_progress(df_edit, sel_month, st.session_state['budgets'])
    if not bp.empty:
        st.dataframe(bp.style.format({
            'limit': '₹{:.0f}', 'used': '₹{:.0f}', 'pct_used': '{:.0%}', 'forecast_eom': '₹{:.0f}'
        }), use_container_width=True)

        # Alerts
        overs = bp[bp['overshoot']]
        if not overs.empty:
            st.warning("Potential overshoot by end of month:")
            st.table(overs[['category', 'limit', 'forecast_eom']].style.format({'limit': '₹{:.0f}', 'forecast_eom': '₹{:.0f}'}))

    # -----------------------------
    # Investment Planner
    # -----------------------------
    st.subheader("Investment Planner (Educational)")
    col1, col2, col3 = st.columns(3)
    with col1:
        profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Growth"], index=1)
        emergency_months = st.slider("Emergency fund target (months of expenses)", 3, 12, 6)
        emergency_current = st.number_input("Current emergency fund (₹)", min_value=0.0, step=1000.0, value=0.0)
    with col2:
        debt_amount = st.number_input("Outstanding high-interest debt (₹)", min_value=0.0, step=1000.0, value=0.0)
        debt_rate = st.number_input("Debt interest rate (annual, e.g., 0.18 for 18%)", min_value=0.0, max_value=1.0, step=0.01, value=0.0)
    with col3:
        emergency_cap_pct = st.slider("Cap emergency allocation (% of surplus)", 0, 100, 30) / 100.0
        debt_cap_pct = st.slider("Cap debt prepayment (% of surplus)", 0, 100, 30) / 100.0

    plan = investment_plan(df_edit, sel_month, emergency_months, emergency_current,
                           debt_amount, debt_rate, profile, emergency_cap_pct, debt_cap_pct)

    st.markdown(f"**Income (₹):** {plan['income']:,.0f} | **Expenses (₹):** {plan['expenses']:,.0f} | **Surplus (₹):** {plan['surplus']:,.0f}")
    if plan['surplus'] <= 0:
        st.error("No investable surplus this month. Review budgets and discretionary spend.")
    else:
        alloc_df = pd.DataFrame({
            'Bucket': list(plan['allocations'].keys()),
            'Amount (₹)': list(plan['allocations'].values())
        })
        st.dataframe(alloc_df, use_container_width=True)

    if plan['notes']:
        with st.expander("Notes & Assumptions"):
            for n in plan['notes']:
                st.write("- ", n)
            st.info("This planner suggests category-level allocations only (Debt/FI, Equity Index, Gold). It does not recommend specific securities or guarantee outcomes. Consider tax implications and consult a licensed advisor.")

    # Q&A
    st.subheader("Ask a question")
    q = st.text_input("Examples: 'Total spending in 2025-10' or 'How much did I spend on Food?' ")
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

st.caption("Local-only prototype. No data leaves your machine.")

