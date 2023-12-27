import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def is_promoted(row_date, promo_dates):
    # row_date (timestamp): date of the row
    # promo_dates (numpy array): an nx2 array that includes the start and end dates of promotions
    for start, end in promo_dates:
        if ((row_date >= start) and (row_date <= end)):
            return True
    return False


def plot_promo_time_series(data):

    d = data.groupby([data['Date'].dt.date])[['SalesQuantity', "IsPromoted"]].sum()
    ax = d[["SalesQuantity"]].plot(figsize=(15, 5))
    d["SalesQuantity"].mask(d['IsPromoted'] == 0, np.nan, inplace=True)
    d["SalesQuantity"].plot(ax=ax, style='-')
    plt.legend(['Not Promoted', 'Promoted'])
    ax.set_title("Sales Quantity (All Products)")
    plt.show()


def generate_avg_weekly_sales(data, by, groups):

    # data (dataframe): train data
    # by (string): either "product" or "store"
    # groups (list): list of product/store codes

    if by == "product":
        field = "ProductCode"
    elif by == "store":
        field = "StoreCode"
    else:
        return None
    
    avg_weekly_sales = {}
    for i in groups:
        d = data[(data[field] == i) & (data["IsPromoted"] == False)]
        d = d.groupby([pd.Grouper(key='Date', freq='W')])['SalesQuantity'].sum()
        avg_weekly_sales[i] = d.values.mean()

    avg_weekly_sales = pd.DataFrame.from_dict(avg_weekly_sales, orient="index", columns=['Avg. Weekly Sales'])
    return(avg_weekly_sales)


def display_clustering_stats(train_data, model):
    # train_data (dataframe): train data, should consist of two columns: the single feature and labels
    # model (kmeans model): the fitted model

    colname = train_data.columns[0]
    labels = np.unique(model.labels_)
    clustering_stats = []
    for label in labels:
        t = train_data[train_data["Label"] == label][colname].describe()
        clustering_stats.append(t)

    clustering_stats = pd.concat(clustering_stats, axis=1)
    clustering_stats.columns = labels
    print(round(clustering_stats, 2))