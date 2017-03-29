from gensim import corpora, models
import gensim
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


stemmer = SnowballStemmer("english")


def read_positive_and_negative_word(data_folder):
    """
    Reads positive and negative adjectives from flat files

    Param_1: File location
    Output_1: Positive words in a list of strings
    Output_2: Negative words in a list of strings
    """
    positiveWords = []
    negativeWords = []
    with open(data_folder+'positive.txt', 'r') as readFile:
        for line in readFile:
            line = line.replace('\n','')
            positiveWords.append(line)

    with open(data_folder+'negative.txt', 'r') as readFile:
        for line in readFile:
            line = line.replace('\n','')
            negativeWords.append(line)
    return (positiveWords, negativeWords)


def filter_positive_negative(book_word_list, positive_words, negative_words):
    """
    Generates a list of positive and negative filtered lists
    
    Param_1: List, containing all words in the book 
    Param_2: List of positive words
    Param_3: List of negative words
    Output_1: List of positive filtered words from all words in the book
    Output_2: List of negative filtered words from all words in the book
    """
    positive_filtered_list = []
    negative_filtered_list = []
    for book in book_word_list:
        current_book_positive_list = []
        current_book_negative_list = []
        for word in book:
            if word in positive_words:
                current_book_positive_list.append(word)
            if word in negative_words:
                current_book_negative_list.append(word)
        positive_filtered_list.append(current_book_positive_list)
        negative_filtered_list.append(current_book_negative_list)
    return (positive_filtered_list, negative_filtered_list)


def lda_topic_modeling(content, topic_count):
    dictionary = corpora.Dictionary(content)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in content]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_count, id2word = dictionary, passes=20)
    #print(ldamodel.print_topics(num_topics=5, num_words=4))
    return lda_model


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
    
def get_similarity_matrix(content_as_str):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2,
                                       stop_words='english',use_idf=True,
                                       tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_as_str) #fit the vectorizer to synopses
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return (similarity_matrix, tfidf_matrix)
