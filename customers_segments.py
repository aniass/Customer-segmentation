"""
Customer segmentation based on RFM method and K-Means clustering.
Model: RFM scoring + K-Means clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

URL = r'Data\sales_data_sample.csv'


def read_data(path):
    '''Read and preprocess data'''
    df = pd.read_csv(path, encoding = 'unicode_escape')
    # Select relevant columns
    col =['CUSTOMERNAME', 'ORDERNUMBER', 'ORDERDATE', 'SALES']
    # Create a new DataFrame with selected columns
    data = df[col]
    # Convert 'ORDERDATE' to datetime
    data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], errors='coerce')
    return data


def create_rfm_table(df, reference_date):
    '''Create the RFM Table: calculate recency, frequency, and monetary value'''
    rfm_df = df.groupby('CUSTOMERNAME').agg({'ORDERDATE': lambda x: (reference_date - x.max()).days,
                                             'ORDERNUMBER': 'count',
                                             'SALES': 'sum'})
    rfm_df.rename(columns={'ORDERDATE': 'Recency', 'ORDERNUMBER': 'Frequency', 'SALES': 'MonetaryValue'}, inplace=True)
    return rfm_df


def calculate_rfm_scores(df):
    '''Calculate RFM scores'''
    # Apply the percentiles of the distribution of the given variable to calculate RFM scores
    r = pd.qcut(df.Recency, 4, labels = list(range(0,4)))
    f = pd.qcut(df.Frequency, 4, labels = list(range(0,4)))
    m = pd.qcut(df.MonetaryValue, 4, labels = list(range(0,4)))
    rfm_df_cutted = pd.DataFrame({'Recency' : r, 'Frequency' : f, 'MonetaryValue' : m})
    return rfm_df_cutted


def show_elbow_plot(rfm_df_cutted):
    '''Display the elbow plot for KMeans clustering'''
    rfm_df_raw = rfm_df_cutted.values
    group = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(rfm_df_raw)
        group.append([i, kmeans.inertia_])
    groups = pd.DataFrame(group, columns = ['number_of_group', 'inertia'])

    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4, style="whitegrid")
    sns.lineplot(data = groups, x = 'number_of_group', y = 'inertia').set(title = "Elbow Method")
    plt.show()


def perform_kmeans_clustering(rfm_df_cutted, rfm_df, n_clusters=4):
    '''KMeans clustering model: perform KMeans clustering on the RFM data'''
    rfm_df_raw = rfm_df_cutted.values 
    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300)
    groups = model.fit_predict(rfm_df_raw)
    rfm_df_cutted['groups'] = groups
    rfm_df['groups'] = groups
    rfm_data = rfm_df
    return rfm_data
    
  
def visualize_groups(df):
    '''Visualize the created groups'''
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Group data by 'groups'
    grouped_data = df.groupby('groups')
    # Plot each group separately
    for i, (group_name, group_data) in enumerate(grouped_data):
        xs = group_data['Recency']
        ys = group_data['MonetaryValue']
        zs = group_data['Frequency']
        ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w', label=group_name)
        
    # Set labels and title
    ax.set_xlabel('Recency')
    ax.set_zlabel('Frequency')
    ax.set_ylabel('MonetaryValue')
    plt.title('Visualization of created groups')
    plt.legend()
    plt.show()


def generate_summary(data):
    '''Generate a summary of the segmentation'''
    # Add 'SegmentName' column
    segment_mapping = {0: 'departing', 1: 'active', 2: 'inactive', 3: 'new'}
    data['SegmentName'] = data['groups'].map(segment_mapping)
    # Display the first few rows of the DataFrame
    print(data.head())
    # Display the distribution of the size of individual groups
    print("\nDistribution of the size of individual groups:")
    print((data.groups.value_counts(normalize = True, sort = True) * 100).to_string())
    # Display statistics for the whole set
    print("\nStatistics for the whole set:")
    print(data.agg(['mean']))
    print("\nThe sum of the values for each group:")
    # Display the sum of the values for each group
    data.groups.value_counts().plot(kind='bar', figsize=(6,4), title='The sum of the values of individual groups')
    plt.show()
    # Save recommendations to a CSV file
    data.to_csv(r'Segmentation_final\customers_segments.csv', index=False)  


if __name__ == '__main__':
    df = read_data(URL)
    reference_date = dt.datetime(2005, 5, 31)  # Define a reference date
    rfm_df = create_rfm_table(df, reference_date)
    rfm_df_cutted = calculate_rfm_scores(rfm_df)
    show_elbow_plot(rfm_df_cutted)
    clustered_data = perform_kmeans_clustering(rfm_df_cutted, rfm_df)
    visualize_groups(clustered_data)
    generate_summary(clustered_data)
  