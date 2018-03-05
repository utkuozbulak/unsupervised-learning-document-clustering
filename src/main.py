from read_and_clean_documents import *
from text_processing import *
from clustering_functions import *
from plot import *
import pandas as pd


DATA_FOLDER = "../data/"
HTML_DATA_FOLDER = "../data/gap-html/"


if __name__ == "__main__":

    # Initial read
    #folder_file_list = get_file_names(HTML_DATA_FOLDER)
    #cleaned_content = get_cleaned_html_documents(folder_file_list)
    #write_list_to_file('cleaned_content.txt', cleaned_content)
    #frequent_words_removed_content_as_list = remove_frequent_items(cleaned_content_as_list, 75)
    #write_list_to_file('freq_words_removed_content.txt', frequent_words_removed_content)

    # Read from cleaned file, not htmls
    (cleaned_content_as_list, cleaned_content_as_str) = \
        read_from_cleaned_file('cleaned_content.txt')
    (frequent_words_removed_content_as_list, frequent_words_removed_content_as_str) = \
        read_from_cleaned_file('freq_words_removed_content.txt')
    (book_names, authors) = read_authors_book_names()

    (similarity_matrix, tfidf_matrix) = get_similarity_matrix(frequent_words_removed_content_as_str)

    km_clusters = get_cluster_kmeans(tfidf_matrix, 5)  # KMeans
    x_pos, y_pos = pca_reduction(similarity_matrix, 10)
    scatter_clusters(x_pos, y_pos, km_clusters, authors) # Scatter K-means with PCA

    dbscan_clusters = get_dbscan_cluster(tfidf_matrix, 1.2)
    dbscan_clusters = dbscan_clusters + 1  # DBScan clusters start from -1
    x_pos, y_pos = multidim_scaling(similarity_matrix, 2)  # MultidimScaling
    scatter_clusters(x_pos, y_pos, dbscan_clusters, authors) # Scatter K-means with PCA

    dendogram(similarity_matrix, book_names)

    lda_model = lda_topic_modeling(frequent_words_removed_content_as_list, 5)
    print(lda_model.print_topics(num_topics=5, num_words=5))
