import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Santander Customer Satisfaction – Advanced Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Santander Customer Satisfaction – Advanced Analytics Dashboard")
st.markdown("---")

# --------------------------------------------------------------------
# GOOGLE DRIVE CSV (HARDCODED)
# --------------------------------------------------------------------
RAW_DRIVE_URL = "https://drive.google.com/file/d/1iN88FzuSEYxsGQK52XFXJgo_VGyIhiRp/view?usp=drive_link"

def convert_drive_url(url):
    file_id = url.split("/d/")[1].split("/")[0]
    return f"https://drive.google.com/uc?export=download&id={file_id}"

CSV_URL = convert_drive_url(RAW_DRIVE_URL)

# --------------------------------------------------------------------
# LOAD & CACHE DATA
# --------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(url):
    df = pd.read_csv(url)
    df = df.dropna(axis=1, how="all")  # clean empty columns
    return df

with st.spinner("Loading large dataset from Google Drive…"):
    df = load_data(CSV_URL)

st.success(f"Loaded dataset: **{df.shape[0]:,} rows × {df.shape[1]:,} columns**")

# --------------------------------------------------------------------
# BASIC METRICS
# --------------------------------------------------------------------
st.header("📊 Dataset Overview")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", f"{len(df):,}")
with col2:
    st.metric("Satisfied (0)", f"{(df['TARGET']==0).sum():,}")
with col3:
    st.metric("Dissatisfied (1)", f"{(df['TARGET']==1).sum():,}")
with col4:
    st.metric("Dissatisfaction Rate", f"{df['TARGET'].mean()*100:.2f}%")

# --------------------------------------------------------------------
# TARGET DISTRIBUTION
# --------------------------------------------------------------------
st.header("🎯 Target Distribution")
st.markdown("---")
target_counts = df['TARGET'].value_counts()

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        x=['Satisfied (0)', 'Dissatisfied (1)'],
        y=target_counts.values,
        labels={'x': 'Class', 'y': 'Count'},
        title="Target Class Counts"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(
        names=['Satisfied', 'Dissatisfied'],
        values=target_counts.values,
        title="Class Imbalance (Pie Chart)"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# CORRELATION HEATMAP (TOP FEATURES)
# --------------------------------------------------------------------
st.header("🔗 Correlation Heatmap (Top 25 Highest-Variance Features)")
st.markdown("---")

# selecting top 25 most variable columns
variance_scores = df.var().sort_values(ascending=False)
top_cols = variance_scores.head(25).index
corr_matrix = df[top_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
st.pyplot(plt.gcf())
plt.clf()

# --------------------------------------------------------------------
# MUTUAL INFORMATION (Feature Importance)
# --------------------------------------------------------------------
st.header("📈 Feature Importance via Mutual Information")
st.markdown("---")

X = df.drop("TARGET", axis=1)
y = df["TARGET"]

@st.cache_data(show_spinner=True)
def compute_mutual_info(X, y):
    scores = mutual_info_classif(X.fillna(0), y)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

mi_scores = compute_mutual_info(X, y)
top10_mi = mi_scores.head(15)

fig = px.bar(
    x=top10_mi.values,
    y=top10_mi.index,
    orientation='h',
    title="Top 15 Features — Mutual Information",
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------------------------
st.header("🌀 PCA Dimensionality Reduction (2D Projection)")
st.markdown("---")

@st.cache_data(show_spinner=True)
def compute_pca(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.drop("TARGET", axis=1))
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    return pcs[:,0], pcs[:,1]

pc1, pc2 = compute_pca(df)

pca_df = pd.DataFrame({
    "PC1": pc1,
    "PC2": pc2,
    "TARGET": df["TARGET"]
})

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="TARGET",
    title="PCA Projection (2D)",
    opacity=0.7
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# K-MEANS CLUSTERING
# --------------------------------------------------------------------
st.header("🔍 K-Means Clustering on PCA Components")
st.markdown("---")

k = st.slider("Choose number of clusters:", 2, 10, 3)

@st.cache_data(show_spinner=True)
def compute_kmeans(df, k):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.drop("TARGET", axis=1))

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(pcs)

    df_temp = pd.DataFrame({
        "PC1": pcs[:,0],
        "PC2": pcs[:,1],
        "PC3": pcs[:,2],
        "Cluster": clusters
    })
    return df_temp

cluster_df = compute_kmeans(df, k)

fig = px.scatter_3d(
    cluster_df,
    x="PC1", y="PC2", z="PC3",
    color="Cluster",
    title=f"3D PCA + KMeans Clustering (k={k})",
    opacity=0.7
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# FEATURE DISTRIBUTIONS
# --------------------------------------------------------------------
st.header("📉 Feature Distribution Explorer")
st.markdown("---")

feature = st.selectbox("Select a feature to visualize:", df.columns)

fig = px.histogram(
    df,
    x=feature,
    color="TARGET",
    nbins=50,
    title=f"Distribution of {feature} by TARGET"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------------------------
st.header("👀 Raw Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Advanced Santander Dashboard — Streamlit</div>",
    unsafe_allow_html=True
)
