# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('Mall_Customers.csv')

dataset = load_data()

# Sidebar Configuration
st.sidebar.title('🏪 KMeans Clustering Customer Segmentation')
st.sidebar.markdown('Gunakan sidebar untuk memilih parameter KMeans')

# Menampilkan 10 baris teratas dari dataset
st.subheader("Sample Data")
st.dataframe(dataset.head(10))

# Statistik dasar dataset
st.subheader("Dataset Statistics")
st.write(dataset.describe())

# Cek Missing Values
st.subheader("Missing Values")
st.write(dataset.isnull().sum())

# Sidebar untuk memilih jumlah cluster
n_clusters = st.sidebar.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=5)

# Fitur yang dipilih untuk clustering
X = dataset.iloc[:, [3, 4]].values

# Menjalankan KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Menampilkan hasil clustering dalam tabel
st.subheader("Clustered Data")
clustered_data = dataset.copy()
clustered_data['Cluster'] = y_kmeans
st.dataframe(clustered_data.head())

# Menampilkan hasil visualisasi elbow method untuk menentukan jumlah cluster
wcss = []
for i in range(1, 11):
    kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans_temp.fit(X)
    wcss.append(kmeans_temp.inertia_)

st.subheader("Elbow Method")
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
st.pyplot(plt)

# Visualisasi Outlier - Boxplot
st.subheader("Outlier Detection - Boxplot")
for column in dataset.select_dtypes(exclude=['object']):
    st.markdown(f'**{column}**')
    plt.figure(figsize=(10, 1.5))
    sns.boxplot(data=dataset, x=column)
    st.pyplot(plt)

# Visualisasi Cluster dalam 2D Scatter plot
st.subheader(f"Visualisasi Clusters dengan {n_clusters} Clusters")
plt.figure(figsize=(10, 6))

colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'gray']
for i in range(n_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
plt.title(f'Clusters of Customers (n_clusters={n_clusters})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
st.pyplot(plt)

# Footer
st.sidebar.markdown("### About")
st.sidebar.info('''
    Aplikasi ini menggunakan KMeans untuk melakukan segmentasi pelanggan berdasarkan *Annual Income* dan *Spending Score*.
    Dataset diambil dari: [Kaggle - Mall Customers Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).
''')
