"""
Simple Santander Customer Satisfaction Dashboard
A minimal version to get started and verify deployment works
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Santander Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Santander Customer Satisfaction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload train.csv", type=['csv'])

if uploaded_file is None:
    st.info("👈 Please upload your train.csv file to begin")
    st.markdown("""
    ### Welcome! 
    
    This dashboard analyzes the Santander Customer Satisfaction dataset.
    
    **To get started:**
    1. Click "Browse files" in the sidebar
    2. Select your train.csv file
    3. Explore the visualizations!
    """)
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

with st.spinner("Loading data..."):
    df = load_data(uploaded_file)

st.success(f"✅ Data loaded! {len(df):,} rows × {len(df.columns):,} columns")

# Basic stats
st.markdown("---")
st.header("📊 Dataset Overview")

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
    dissatisfaction_rate = df['TARGET'].mean() * 100
    st.metric("Dissatisfaction Rate", f"{dissatisfaction_rate:.2f}%")

# Target Distribution
st.markdown("---")
st.header("🎯 Customer Satisfaction Distribution")

col1, col2 = st.columns(2)

with col1:
    # Bar chart
    target_counts = df['TARGET'].value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=['Satisfied (0)', 'Dissatisfied (1)'],
            y=target_counts.values,
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f'{v:,}' for v in target_counts.values],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title='Count by Satisfaction',
        yaxis_title='Number of Customers',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=['Satisfied', 'Dissatisfied'],
            values=target_counts.values,
            marker_colors=['#1f77b4', '#ff7f0e'],
            hole=0.4
        )
    ])
    fig.update_layout(
        title='Satisfaction Distribution',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Key Insights
st.markdown("---")
st.header("💡 Key Insights")

satisfied_pct = (df['TARGET'] == 0).mean() * 100
dissatisfied_pct = (df['TARGET'] == 1).mean() * 100
imbalance_ratio = satisfied / dissatisfied

st.markdown(f"""
- **Class Distribution**: {satisfied_pct:.1f}% satisfied vs {dissatisfied_pct:.1f}% dissatisfied
- **Imbalance Ratio**: {imbalance_ratio:.1f}:1 (highly imbalanced dataset)
- **Total Features**: {len(df.columns) - 1} predictive features
- **Missing Values**: {df.isnull().sum().sum()} ({(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%)
""")

# Correlation Analysis
st.markdown("---")
st.header("🔗 Top Correlated Features")

with st.spinner("Calculating correlations..."):
    correlations = df.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)
    
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Top 10 Positive Correlations")
    top_positive = correlations.head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_positive.index,
            x=top_positive.values,
            orientation='h',
            marker_color='#2ca02c'
        )
    ])
    fig.update_layout(
        xaxis_title='Correlation with Dissatisfaction',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📉 Top 10 Negative Correlations")
    top_negative = correlations.tail(10)
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_negative.index,
            x=top_negative.values,
            orientation='h',
            marker_color='#d62728'
        )
    ])
    fig.update_layout(
        xaxis_title='Correlation with Dissatisfaction',
        height=400,
        yaxis={'categoryorder': 'total descending'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Feature Statistics
st.markdown("---")
st.header("📊 Feature Statistics")

# Show correlation table
st.subheader("Top 20 Features by Absolute Correlation")
top_20_abs = correlations.abs().sort_values(ascending=False).head(20)

df_display = pd.DataFrame({
    'Feature': top_20_abs.index,
    'Correlation': [correlations[f] for f in top_20_abs.index],
    'Abs Correlation': top_20_abs.values,
    'Direction': ['↑ Risk Factor' if correlations[f] > 0 else '↓ Protective Factor' 
                  for f in top_20_abs.index]
})

st.dataframe(
    df_display.style.format({
        'Correlation': '{:+.4f}',
        'Abs Correlation': '{:.4f}'
    }),
    use_container_width=True,
    height=400
)

# Data Preview
st.markdown("---")
st.header("👀 Data Preview")

st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset Info")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")
    
with col2:
    st.metric("Columns", f"{df.shape[1]:,}")
    
with col3:
    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Santander Customer Satisfaction Dashboard v1.0</p>
    <p>Built with Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
