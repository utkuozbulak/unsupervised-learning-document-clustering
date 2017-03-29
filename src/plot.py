import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.cluster.hierarchy import ward, dendrogram


def scatter_clusters(x_pos, y_pos, clusters, titles):
    cluster_colors = {0: '#cc0000',
                      1: '#006600',
                      2: '#002699',
                      3: '#ffff33',
                      4: '#ffa64d',
                      5: '#000000'}
    # As many as items
    cluster_names = {0: '',
                 1: '',  
                 2: '', 
                 3: '',
                 4: '',
                 5: ''}
                 
    df = pd.DataFrame(dict(x= x_pos, y= y_pos, label= clusters, title= titles)) 
    groups = df.groupby('label')
    fig, ax = plt.subplots(figsize=(17, 9))  # Set size
    ax.set_axis_bgcolor('#e6f7ff')
    # Iterate through groups to layer the plot
    for name, group in groups:
        ax.plot(group.x, group.y, marker='D', linestyle='solid', ms=15, 
                label=cluster_names[name], color=cluster_colors[name], mec='black')
        ax.set_aspect('auto')
        ax.tick_params(axis= 'x', which='both', labelbottom='off')
        ax.tick_params(axis= 'y', which='both', labelleft='off')
    ax.legend(numpoints=1)

    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size= 15)  
    plt.show() # Show the plot


def dendogram(similarity_matrix, book_names):
    linkage_matrix = ward(similarity_matrix) # Define the linkage_matrix using ward clustering pre-computed distances
    mpl.rcParams['lines.linewidth'] = 5

    fig, ax = plt.subplots(figsize=(15, 20)) # Set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=book_names);

    plt.tick_params(\
        axis= 'x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off',
        length = 25)
    plt.tick_params(\
        axis= 'y',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off',
        labelsize = 20)
    plt.tick_params(width=50, length = 10)
    plt.tight_layout() # Show plot with tight layout