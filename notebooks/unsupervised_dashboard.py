import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
import dash
from dash import dcc, html, Input, Output
import plotly.express as px


# Load processed data - preprocessed_data.csv. Adjust path as needed, e.g., "/Users/yourusername/project/preprocessed_data.csv".
windsor = pd.read_csv('/Users/../preprocessed_data.csv', index_col=False)


dataset = windsor.copy()

def kmeans(dataset, n_clusters, random_state = 42, n_init = 'auto'):
    """
    Function: to perform k-means clustering on the provided dataset.

    Parameters:
    dataset (DataFrame): The dataset to cluster.
    n_clusters (int): The number of clusters.
    random_state (int): The random state for reproducibility.
    n_init (int or str): Number of times the k-means algorithm will be run with different centroid seeds.

    Returns:
    labels, inertia, and silhouette score.
    """
    kmeans = KMeans(n_clusters= n_clusters, random_state = random_state, n_init=n_init)
    kmeans.fit(dataset)
    klabels = kmeans.labels_
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(dataset, klabels)

    return klabels, inertia, silhouette_avg

klabels, inertia, silhouette_avg = kmeans(windsor, 2, random_state = 42, n_init = 'auto')


# Dimension reduction using TSNE algorithm(algo)
def tsne_data(dataset, n_components, random_state = 42):
    tsne = TSNE(n_components = n_components, random_state=random_state)
    tsne_data = tsne.fit_transform(dataset)

    return tsne_data

tsne_data_2d = tsne_data(windsor, 2, random_state = 42) # Dimension reduction to 2 dimensions
tsne_data_3d = tsne_data(windsor, 3, random_state = 42) # Dimension reduction to 3 dimensions

# Dimension reduction using PCA algorithm(algo)
def pca_data(dataset, n_components):
    pca = PCA(n_components= n_components)
    pca_data = pca.fit_transform(dataset)

    return pca_data

pca_data_2d =  pca_data(windsor, 2) # Dimension reduction to 2 dimensions
pca_data_3d =  pca_data(windsor, 3) # Dimension reduction to 3 dimensions

# Creating an interactive clustering dashboard using Dash and Plotly ( TSNE and PCA algorithms in 2D and 3D )

app = dash.Dash(__name__)

# Load processed data - preprocessed_data.csv. Adjust path as needed, e.g., "/Users/yourusername/project/preprocessed_data.csv".
windsor = pd.read_csv('/Users/../preprocessed_data.csv', index_col=False)

dataset = windsor.copy()

app.layout = html.Div([
    html.H1("Interactive Clustering Dashboard: K-means in 2D and 3D with TSNE and PCA",style ={'textAlign':'center'}),
    
    dcc.RadioItems(id= 'slct_dmnsn_rdctn_algo', options= [{'label':'TSNE', 'value':'TSNE'},{'label':'PCA', 'value':'PCA'}],value= 'TSNE'),
    dcc.RadioItems(id= 'slct_dmnsn', options = [{'label': '2D', 'value': 2}, {'label':'3D', 'value': 3}], value= 2),
    
    html.Br(),

    html.H2("Clustering Performance Metrics", style={'textAlign': 'center'}),
    html.P(f"Inertia: {inertia:.2f}"),
    html.P(f"Silhouette Score: {silhouette_avg:.2f}"),

dcc.Graph(id ='my_Graph'),

])

@app.callback(
    Output(component_id= 'my_Graph', component_property= 'figure'),
    [Input(component_id= 'slct_dmnsn_rdctn_algo', component_property= 'value'),
     Input(component_id= 'slct_dmnsn', component_property= 'value')]

    )
def update_graph(algo, dimension):
    #print(algo, dimension)
    #print(type(algo), type(dimension))

    dataset = windsor.copy()

    if algo == 'TSNE':
        data = tsne_data_2d if dimension == 2 else tsne_data_3d
    else: # if algo == 'PCA'
        data =  pca_data_2d if dimension == 2 else pca_data_3d   

    if dimension == 2:
        fig = px.scatter(
            x= data[:, 0],
            y= data[:,1],
            color = klabels,
            color_continuous_scale= 'tropic',
            title = f"{algo} {dimension}D Visualisation",
            labels= {'x':'Component 1', 'y':'Component 2'}
            
        )  
    else: #  dimension == 3
        fig = px.scatter_3d(
            x= data[:, 0],
            y= data[:, 1],
            z= data[:, 2],
            color= klabels,
            color_continuous_scale= 'tropic',
            title= f"{algo} {dimension}D Visualisation" ,
            labels= {'x':'Component 1', 'y':'Component 2', 'z':'Component 3'}
        )
    return fig

if __name__ == '__main__':
    app.run_server(debug= True)    