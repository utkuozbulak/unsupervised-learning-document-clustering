import glob
import re
from html.parser import HTMLParser
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.porter import PorterStemmer


directory = '../data/cleaned_data/'


def read_authors_book_names():
    with open(directory + 'authors.txt', "r") as f:
        authors = []
        for line in f:
            line_str = line.strip()
            authors.append(line_str)
    f.close()

    with open(directory + 'books.txt', "r") as f:
        books = []
        for line in f:
            line_str = line.strip()
            books.append(line_str)
    f.close()
    return (books, authors)

def write_list_to_file(file_name, content):
    f = open(directory + file_name, 'a')
    for list in content:
        for item in list:
            f.write(str(item))
            f.write(' ')
        f.write('\n')
    f.close()


def read_from_cleaned_file(file_name):
    with open(directory + file_name, "r") as f:
        content_as_list = []
        content_as_str = []
        for line in f:
            line_str = line.strip()
            line_list = line_str.split()
            content_as_list.append(line_list)
            content_as_str.append(line_str)
    f.close()
    return (content_as_list, content_as_str)


def read_word_list(file_name):
    word_list = []
    with open(directory + file_name, "r") as f:
        for line in f:
            line_str = line.strip()
            line_list = line_str.split()
            content_as_list.append(line_list)
            content_as_str.append(line_str)
    f.close()
    return (content_as_list, content_as_str)


class MyHTMLParser(HTMLParser):  # To parse html files
    html_plain_document = []
    def handle_data(self, data):  # For every html data
        self.html_plain_document.append(data)  # Append the data to the list


def get_file_names(data_folder):
    """
    Creates a list of list containing folder and file names

    Param_1: Data folder name as string
    Output_1: List of lists containng folder and file names
    """
    folder_list = glob.glob(data_folder+"*")  # Get folder and file names names in a list
    folder_file_list = []
    for index,folder in enumerate(folder_list):
        folder_file_list.append([])
        folder_name = [ folder[folder.rfind('/') + 1 : ] ]  # Get folder name
        file_names = glob.glob(folder+ '/*.html')  # Get html file names in a list
        file_names = sorted(file_names)
        folder_file_list[index].append(folder_name)
        folder_file_list[index].append(file_names)
    return folder_file_list


def get_cleaned_html_documents(folder_file_list):
    """
    Reads the files in folders
    Returns a list of list containing cleaned words/data in the html files

    Param_1: Output of get_file_names, list of folders and file names
    Output_1: List of lists containing string, which are words in the html
    """
    html_parser = MyHTMLParser()
    cleaned_documents = []
    for item in folder_file_list:
        for file_name in item[1]:
            f = open(file_name, "r")  # Open the file
            x = f.read()  # Read file
            html_parser.feed(x)  # Feed the file to get rid of html elements
            f.close()  # Close the file
        plain_html_file = html_parser.html_plain_document  # Get the list of words
        cleaned_html_file = clean_list(plain_html_file)  # Clean the list
        cleaned_documents.append(cleaned_html_file)
        html_parser.html_plain_document = []  # Empty the html document
    return cleaned_documents


def clean_list(list_to_clean):
    """
    Function to clean a list
    Removes any non-alphanumeric characters
    Stems words
    Gets rid of any empty elements in the list

    Param_1: List, containing strings
    Output_1: List, containing cleaned strings
    """
    stemmer = PorterStemmer()
    items_to_clean = set(list(stopwords.words('english')) + ['\n','\n\n','\n\n\n','\n\n\n\n','ocroutput','',' '])
    # Items to clean
    regex_non_alphanumeric = re.compile('[^0-9a-zA-Z]')  # REGEX for non alphanumeric chars
    for index,item in enumerate(list_to_clean):
        item = regex_non_alphanumeric.sub('', item)  # Filter text, remove non alphanumeric chars
        item = item.lower()  # Lowercase the text
        item = stemmer.stem(item)  # Stem the text
        if len(item) < 3:  # If the length of item is lower than 3, remove item
            item = ''
        list_to_clean[index] = item  # Put item back to the list
    cleaned_list = [elem for elem in list_to_clean if elem not in items_to_clean]
    # Remove empty items from the list
    return cleaned_list


def remove_frequent_items(book_word_list, percentage):
    """
    Remove frequently occured words

    Param_1: List of list containing strings
    Param_2: Above x percentage of occurance will be removed
    Output_1: Cleaned list
    """
    treshold = int(len(book_word_list) * percentage / 100)
    DF = defaultdict(int)
    for cleaned_list in book_word_list:
        for word in set(cleaned_list):
                DF[word] += 1
    words_to_remove = {k:v for k,v in DF.items() if v > treshold }
    # A new dictionary of items that only has count above treshold
    words_to_remove_as_list = set(words_to_remove.keys())
    freq_items_removed_book_word_list = []
    for book in book_word_list:
        freq_items_removed_list = [word for word in book if word not in words_to_remove_as_list]
        freq_items_removed_book_word_list.append(freq_items_removed_list)
    return freq_items_removed_book_word_list


def convert_list(content):
    converted_content = []
    for index, str_list in enumerate(content):
        converted_content.append('')
        for item in str_list:
            converted_content[index] = converted_content[index] + ' ' + item
    return converted_content
