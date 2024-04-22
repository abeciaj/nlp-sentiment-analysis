# import libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# download nltk corpus (first time only)
import nltk

nltk.download('all')

# Load the amazon review dataset
# https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv
df = pd.read_csv('<csv_file>') 

# Todo: select a category and create a csv file from the positive and negative category. 
# https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

df

# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# apply the function df
df['reviewText'] = df['reviewText'].apply(preprocess_text)
df

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# create get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment

# apply get_sentiment function
df['sentiment'] = df['reviewText'].apply(get_sentiment)
df

from sklearn.metrics import confusion_matrix
print(confusion_matrix(df['Positive'], df['sentiment']))

from sklearn.metrics import classification_report
print(classification_report(df['Positive'], df['sentiment']))