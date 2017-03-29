from string import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sns.set(style="white")
    sns.set(font_scale=1)
    mask = np.zeros_like(similarity_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = False
    f, ax = plt.subplots(figsize=(11, 9))
    
    c = sns.heatmap(b, mask=mask, cmap=cmap, vmax=.8,
                square=True, linewidths=0.01,  ax=ax)
    c.set(xlabel='Document ID', ylabel='Document ID',fontsize=25)
    plt.show()
    fig = c.get_figure()
    fig.suptitle('TF-IDF Document Similarity Matrix', fontsize=25)
    
    fig.savefig("output.png")
