#-----
# SMS Spam Classification
# Author: Sarah H
# Date: 13 Apr 2021
#-----

# Import libraries ---------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re
#pip install autocorrect
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Import data --------------------------------------------------------------------
df = pd.read_csv("https://raw.githubusercontent.com/SarahHannes/ml/main/datasets/naive_bayes_spam/spam.csv")

# Initial pre-processing ---------------------------------------------------------
# 1: Remove redundant columns
df = df[['v1', 'v2']]

# 2: One hot encoding for ham/ spam label (ham = 0; spam = 1)
label = LabelEncoder()
label.fit(['ham', 'spam'])
df['v1'] = label.transform(df[['v1']])

# 3: Renaming columns
df.rename(columns = {'v1':'label', 'v2':'text'}, inplace=True)

# 4: Adding a new column for character length
df.insert(2, 'length', df['text'].apply(len))

# 5: Adding a new column for number of words count
df.insert(3, 'num_words', 
          df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1).apply(len))

# 6: Adding a new column for number of sentences count
df.insert(4, 'num_sentences',
         df.apply(lambda row: nltk.sent_tokenize(row['text']), axis=1).apply(len))

# Summary
print('Overall summary')
print(df.describe().T)
print('----------------------')
print('Summary by class label')
print(df.groupby('label').describe().T)

# Exploratory Data Analysis ---------------------------------------------------------
# 1: Bar chart for count of observations
cols = {
    0:'#37a9b7',
    1:'#fd6467'}

plt.figure(figsize=(15,10))
fig = sns.countplot(x=df['label'], orient='h', palette=cols)
fig.set_title('Count of class labels')
plt.show(fig)

# 2: Histogram to look at the overall distribution
plt.figure(figsize=(15, 10))
plt.hist(x=df['length'], bins='auto', rwidth=0.5)
plt.title('Histogram showing distribution of character length')
plt.xlabel('Character length')
plt.ylabel('Frequency')
plt.show()

# 3: Histogram separating different classes
df.hist(column='length', by='label', bins=30, figsize=(15,8))

# 4: Pairplots to look at relationships between features
# Original plots
plt.figure(figsize=(30,35))
fig = sns.pairplot(data=df, hue='label', palette=cols)
plt.show(fig)

# Zooming in
plt.figure(figsize=(30,35))
fig = sns.pairplot(data=df, hue='label', palette=cols)

# 1st row 1st col
fig.axes[0,0].set_xlim((0,200))

# 2nd row 2nd col
fig.axes[1,1].set_xlim((0,60))

# 3rd row 3rd col
fig.axes[2,2].set_xlim((0,10))

plt.show(fig)

# Additional pre-processing ---------------------------------------------------------
# currently spellcheck() is not in use - operation time takes too long (not sure why)

def spellcheck(text):
    """
    input: string
    Corrects mispelt words.
    return: string
    """
    #spell = Speller()
    return spell(text)

def remove_stopword(text):
    """
    input: string
    Remove stopwords.
    return: string without stopwords    
    """
    filtered = []
    #stopword = set(stopwords.words('english'))
    # for every word in text, append it to filtered list if the word is not in stopword set
    for word in text.split(' '):
        if word not in stopword:
            filtered.append(word)
    return ' '.join(filtered)

def get_clean_text(text):
    """
    input: string
    Convert to lowercase, remove all punctuations, remove stopwords, correct spellings.
    return: string
    """
    # do spellcheck
    #text = spellcheck(text) #operation takes too long.
    # remove stopword
    text = remove_stopword(text)
    # strip all characters other than alphanumeric, make all lowercase, trim spaces before and after
    #t = text.translate(str.maketrans("","", string.punctuation)).strip().lower() #operation takes too long.
    t = re.sub('[^a-zA-Z]', " ", text) # remove all except alphabets
    # remove double/ multiple whitespaces
    t = re.sub("\s\s+", " ", t)
    return t


def lemmatize_word(text):
    """
    input: string
    lemmatizes words. (eg teaches -> teach)
    output: string
    """
    t = []
    #wnl = WordNetLemmatizer()
    for word in text.split(' '):
        t.append(wnl.lemmatize(word))
    return " ".join(t)

def get_stem(text):
    """
    input: string
    return word stem using SnowballStemmer, ignore stopwords. (eg sailed -> sail)
    return: string
    """
    t = []
    #stemmer = SnowballStemmer('english', ignore_stopwords = True)
    for word in text.split(' '):
        t.append(stemmer.stem(word))
    return " ".join(t)

# Initializing instances
#spell = Speller()
stemmer = SnowballStemmer('english', ignore_stopwords = True)
wnl = WordNetLemmatizer()
stopword = set(stopwords.words('english'))

# 1: Adding a new column for cleaned text
df['cleaned_text'] = df['text'].apply(get_clean_text)

# 2: Adding a new column for lemmatized text
df['lemmatize_text'] = df['cleaned_text'].apply(lemmatize_word)

# 3: Adding a new column for stemmed text
df['stem_text'] = df['lemmatize_text'].apply(get_stem)

# Visualization ----------------------------------------------------------------------------

# 1: Filter only the processed text for Spam and Ham
spams = df[df['label']==1].stem_text
hams = df[df['label']==0].stem_text

# 2: Compile all in separate corpora
spam_list = []
for row in spams:
    spam_list.append(row)
    
spam_c = ' '.join([str(s) for s in spam_list])

ham_list = []
for row in hams:
    ham_list.append(row)
    
ham_c = ' '.join([str(h) for h in ham_list])

# 3: Generate wordclouds
spam_wc = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(spam_c)

ham_wc = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(ham_c)

# 4: Visualizing wordclouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
#fig.suptitle('Wordclouds')
ax1.imshow(spam_wc)
ax2.imshow(ham_wc)
ax1.title.set_text('Spams')
ax2.title.set_text('Hams')
ax1.axis('off')
ax2.axis('off')
plt.show()

# Vectorization & Train/ Test split ---------------------------------------------------------

# 1: Create a corpus of all of our processed texts
corpus = []
for row in df['stem_text']:
    corpus.append(row)

# 2: Vectorization using TF-IDF & assigning it as our X variable
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
# X.shape # examining the shape (5572, 6305)

# 3: Assign y variables
y = df['label']

# 4: Split test/ train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model training ----------------------------------------------------------------------------

# 1: Initializing & fitting into naive bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# 2: Predicting
y_hat = nb.predict(X_test)
y_hat_prob = nb.predict(X_test) # to get probability estimate
print(y_hat)
print(y_hat_prob)

# Model evaluation ----------------------------------------------------------------------------

# 1: Classification Report
target_name = ['ham', 'spam']
print(metrics.classification_report(y_test, y_hat, target_names=target_name))

# 2: Log loss
print(f'Log loss: {metrics.log_loss(y_test, y_hat_prob):.3f}')
