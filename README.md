# Customer segmentation

### The customer segmentation based on RFM method and K-Means clustering

## General info
The project contains two methods of customer segmentation by combining the RFM method and K-Means clustering. The dataset includes sample sales data based on retail analytics. The analysis was performed in two approaches:
- **the first** one uses RFM scoring (assigned RFM score) and K-means clustering;
- **the second** raw calculated RFM variables and K-means clustering.

### Dataset
The dataset includes sample sales data based on retail analytics and contains three years of sales. It comes from Kaggle and can be find [here](https://www.kaggle.com/kyanyoga/sample-sales-data).

## Motivation
The customer segmentation is an effective method that enables to get better know clients and to better correspond their various needs. 
Almost every company that sells products or services stores data of shopping. This type of data can be used to execute customer segmentation thus the results of the analysis can be translated into marketing campaigns to increase sales. One of the most widely used techniques is RFM analysis, which allows to create personalized special offers to improve sales. 

**RFM** stands for Recency, Frequency, Monetary Value and it is the technique of customer segmentation based on their transaction history. The RFM analysis is based on three criterias which measure different customer characteristics:
- Recency: Days since last purchase/order of the client;
- Frequency: Total number of purchases the customer were made;
- Monetary Value: Total money the customer spent per order.

## Project contains:
- First approach - **Customer_segmentation.ipynb**
- Second approach - **Segmentation_Kmeans.ipynb**
- Python script with customer segmentation - **customers_segments.py**

## Technologies

The project is created with:

- Python 3.6
- libraries: pandas, numpy, sklearn, scipy, seaborn, matplotlib.

**Running the project:**

To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    customers_segments.py
