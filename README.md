# Document Clustering with Python

This is a project to apply document clustering techniques using Python. For this exercise, we started out with texts of 24 books taken from Google as part of Google Library Project. 

## Data Cleaning

The files were read using an OCR system and contained HTML tags all over the place so the first step before starting the clustering was data cleaning. Data cleaning process was like below:

1- Get rid of HTML Tags (with Python HTMLParser Library)

2- Remove punctuations

3- Remove non-English characters

4- Lowercase all the words

5- Stem the words using Porter Stemmer

6- Remove all stop words

7- Remove high frequency words ( Words with 75%+ occurance rate in all books )

## Document Similarity

After cleaning the data, next step is to find the document similarity matrix and sparse document vectors using TF-IDF. Below is the representation of of similarity matrix. 

![TFIDF](https://raw.githubusercontent.com/utkuozbulak/document-clustering/master/plots/similarity_matrix.png "TFIDF")

## Clustering and Dimensionality Reduction

After obtaining similarity matrix and sparse vectors of documents from TF-IDF, we started applying clustering techniques and used dimensionality reduction techniques to be able to visualise it in 2D.


### K-Means and Principle Component Analysis

K-means clustering is one of the popular clustering techniques, with K=5 and PCA dimensioanlity reduction, it generated following output.

![KPCA](https://raw.githubusercontent.com/utkuozbulak/document-clustering/master/plots/lined_2.png "KPCA")

### DBScan and Multidimensional Scaling

DBScan is yet another clustering algorithm we can use to cluster the documents. With epsilon value 1.2, it generates 4 clusters and if we combine it with MDS, it generates following output.

![DBSMDS](https://raw.githubusercontent.com/utkuozbulak/document-clustering/master/plots/dbscan-mds.png "DBSMDS")


### Hierarchical Clustering with WARD

WARD's method is commonly used to generate hierarchical clusters, below is the generated hierarchical clustering plot if we apply it to our documents.

![WARD](https://raw.githubusercontent.com/utkuozbulak/document-clustering/master/plots/hier_clustering.png "WARD")

## Package Info

/src/ contains all the code to generate the plots

/cleaned_data/ contains cleaned data (The data after the cleaning step)

/plots/ contains generated images



