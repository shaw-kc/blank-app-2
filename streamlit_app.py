import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------
# Page Setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="Santander Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Santander Customer Satisfaction Dashboard")
st.markdown("---")

# ----------------------------------------------------------
# Google Drive CSV Loader
# ----------------------------------------------------------
st.sidebar.header("📁 Data Source")

drive_url = st.sidebar.text_input(
    "Paste Google Drive CSV Link (Share → Copy link)",
    placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
)

def convert_drive_url(url):
    """Convert Google Drive share link to direct-download CSV link."""
    if "drive.google.com" not in url:
        return None
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    except:
        return None

@st.cache_data
def load_data(link):
    return pd.read_csv(link)

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------
if not drive_url:
    st.info("👈 Enter your Google Drive CSV link to load the dataset.")
    st.stop()

converted_link = convert_drive_url(drive_url)

if converted_link is None:
    st.error("❌ Invalid Google Drive link format. Make sure it contains `/d/FILE_ID/`")
    st.stop()

try:
    with st.spinner("Loading data from Google Drive..."):
        df = load_data(converted_link)
    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns):,} columns")
except Exception as e:
    st.error("❌ Failed to load CSV file. Check permissions or link.")
    st.exception(e)
    st.stop()

# ----------------------------------------------------------
# BASIC METRICS
# ----------------------------------------------------------
st.header("📊 Dataset Overview")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", f"{len(df):,}")

with col2:
    satisfied = (df['TARGET'] == 0).sum()
    st.metric("Satisfied", f"{satisfied:,}")

with col3:
    dissatisfied = (df['TARGET'] == 1).sum()
    st.metric("Dissatisfied", f"{dissatisfied:,}")

with col4:
    st.metric("Dissatisfaction Rate", f"{df['TARGET'].mean() * 100:.2f}%")

# ----------------------------------------------------------
# TARGET DISTRIBUTION
# ----------------------------------------------------------
st.header("🎯 Customer Satisfaction Distribution")
st.markdown("---")

target_counts = df['TARGET'].value_counts()

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        target_counts,
        x=target_counts.index,
        y=target_counts.values,
        labels={'x': 'Target', 'y': 'Count'},
        title="Count by Satisfaction"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(
        values=target_counts.values,
        names=["Satisfied (0)", "Dissatisfied (1)"],
        title="Satisfaction Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# CORRELATIONS
# ----------------------------------------------------------
st.header("🔗 Top Correlated Features")
st.markdown("---")

correlations = df.corr(numeric_only=True)['TARGET'].drop('TARGET')
correlations_sorted = correlations.sort_values(ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Top 10 Positive Correlations")
    top_pos = correlations_sorted.head(10)
    st.bar_chart(top_pos)

with col2:
    st.subheader("📉 Top 10 Negative Correlations")
    top_neg = correlations_sorted.tail(10)
    st.bar_chart(top_neg)

# ----------------------------------------------------------
# DATA PREVIEW
# ----------------------------------------------------------
st.header("👀 Data Preview")
st.markdown("---")

st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset Info")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Memory Usage (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Santander Dashboard v1.0 — Streamlit</div>",
    unsafe_allow_html=True
)
