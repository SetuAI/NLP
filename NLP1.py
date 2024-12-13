
"""
Spacy tokenization

splitting text into meaninful sentences and words. (sentence tokenization and word tokenization)


now in sentence tokenization, we need language understanding

for ex, Dr. means Doctor and does not mean sentence ends after Dr.

Example 2: "Let's go to New York !."

"token 1" : "
"token 2" : Let
"token 3": 's
"token 4" : go
"token 5" : to
"token 6" : New
"token 7" : York
"token 8" : !
"token 9" : "


"""

#%%


import spacy
"""
#nlp : language object
#"en" : english 
"de" : german

"""
nlp = spacy.blank("en")

doc = nlp("Musk co-founded the AI research organization OpenAI with Sam Altman in 2015. Musk left the company's board in 2018, saying of his decision that he \
          didn't agree with some of what OpenAI team wanted to do\
          OpenAI went on to launch ChatGPT in 2022, and GPT-4 in March 2023. \
          that month, Elon Musk was one of the individuals to sign an open letter\
          from the Future of Life Institute calling for a six-month pause in the development of any AI software more powerful than GPT-4")

for token in doc:
    print(token,end=' ')
    #print(len(token))
    
"to grab index : you can use list like operations"

doc[0]
doc[-1]

doc[0:5]

token3 = doc[3]
token3.text

dir(token3)

token3.is_alpha #is token an alphabet

token3.like_num #is token a number

#%%


# example

ex = nlp("Tony gave two $ to Peter.")
print(ex)

for token in enumerate(ex):
    print(token)
    
ex[2].is_alpha #is token 2 in this ex[2] alphbet ? 

ex[3].is_currency  # is token 3 currency


"create look to check how many alpha, currency, etc are there in the context"

for token in doc:
    "token.i stores index "
    print(token, "==>", "index: ", token.i, "is_alpha:", token.is_alpha, 
          "is_punct:", token.is_punct, 
          "like_num:", token.like_num,
          "is_currency:", token.is_currency,
         )

#%%

text='''
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.
gov/, and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/, 
and the European Social Survey at http://www.europeansocialsurvey.org/.
'''


doc = nlp(text)
data_websites = [token.text for token in doc if token.like_url ] 
data_websites


"Figure out all transactions from this text with amount and currency"

transactions = "Tony gave two $ to Peter, Bruce gave 500 € to Steve"
doc = nlp(transactions)
for token in doc:
    if token.like_num and doc[token.i+1].is_currency:
        print(token.text, doc[token.i+1].text) 



#%%

"""
STOP WORDS

In natural language processing (NLP), stop words are commonly occurring words that are often removed from text during preprocessing because they're considered to have little or no semantic value for many NLP tasks.


Common examples: Words like "the," "a," "an," "is," "are," "was," "were," "in,"
 "on," "at," "to," "of," "for," "with," "by," "from," "as," "it," "they," "them",
 "he," "she," "his," "her," "you," "me," "my," "I," "we," "our," and so on are often considered stop words.

"""

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

len(STOP_WORDS)

nlp = spacy.load("en_core_web_sm")

doc = nlp("We just opened our wings, the flying part is coming soon")

stopwords=[]

for token in doc:
    if token.is_stop:
        stopwords.append(token)
print(stopwords)

        

def preprocess(text):
    "#if not a stopword add it to the no_stop_words list"
    doc = nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop]
    return " ".join(no_stop_words) 

preprocess("In short, stop words are frequent words that are often removed in NLP preprocessing to improve efficiency and potentially accuracy. ")

preprocess("The other is not other but your divine brother")



#%%

"""
Stemming and Lemmatization : preprocessing stage

in you google search result, you might have seen you get some matching words 
in the output as well

get base word : stemming

talking - talk
running - run
adjustable - adjust

The base word that you get is called "lemma"

eat is the lemma for ate : this is called lemmatization

Stemming : using fixed rules to get to the base word 

Lemmatization : Use linguistic knowledge to derive a base word

There is a major overlap in between the two 

"""

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

words = ["eating", "eats", "eat", "ate", "adjustable",
         "rafting", "ability", "meeting"]

for word in words:
    print(word, "|", stemmer.stem(word))
    

#%%

#let import necessary libraries and create the object

#for nltk
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

#downloading all neccessary packages related to nltk
nltk.download('all')


#for spacy
import spacy
nlp = spacy.load("en_core_web_sm")

"""
spacy.load(): This function is used to load a trained spaCy pipeline. A pipeline is a sequence of processing components that analyze text.

"en_core_web_sm": This string specifies the name of the model you want to load. It refers to:

en: English language.

core: A general-purpose model (as opposed to a model trained for a specific task).

web: Indicates the model is trained on web text.

sm: Stands for "small". This is a smaller, faster model, suitable for many common NLP tasks.

"""
    
"""
Convert these list of words into base form using Stemming and Lemmatization and observe the transformations
Write a short note on the words that have different base words using stemming and Lemmatization

"""

#using stemming in nltk
lst_words = ['running', 'painting', 'walking', 'dressing', 'likely', 'children', 'whom', 'good', 'ate', 'fishing']

for word in lst_words:
    print(f"{word} | {stemmer.stem(word)}")
    


#using lemmatization in spacy

doc = nlp("running painting walking dressing likely children whom good ate fishing")
for token in doc:
    print(token, " | ", token.lemma_)
    
    
"""
Observations

Words that are different in stemming and lemmatization are:

painting
likely
children
ate
fishing
As Stemming achieves the base word by removing the suffixes [ing, ly etc], so it successfully\ 
 
 transform the words like 'painting', 'likely', 'fishing' and lemmatization fails for some words ending with suffixes here.

As Lemmatization uses the dictionary meanings while converting to the base form, 

so words like 'children' and 'ate' are successfully transformed and stemming fails here.

"""
#%%

"convert the given text into it's base form using both stemming and lemmatization"

text = """Latha is very multi talented girl.She is good at many skills like dancing, running, singing, playing.She also likes eating Pav Bhagi. she has a 
habit of fishing and swimming too.Besides all this, she is a wonderful at cooking too.
"""

#using stemming in nltk

#step1: Word tokenizing
all_word_tokens = nltk.word_tokenize(text)


#step2: getting the base form for each token using stemmer
all_base_words = []

for token in all_word_tokens:
  base_form = stemmer.stem(token)
  all_base_words.append(base_form)


#step3: joining all words in a list into string using 'join()'
final_base_text = ' '.join(all_base_words)
print(final_base_text)

"---------------------------------------------------------------------------"
#using lemmatisation in spacy


#step1: Creating the object for the given text
doc = nlp(text)
all_base_words = []

#step2: getting the base form for each token using spacy 'lemma_'
for token in doc:
  base_word =  token.lemma_
  all_base_words.append(base_word)


#step3: joining all words in a list into string using 'join()'
final_base_text = ' '.join(all_base_words)

print(doc)
print("\n")
print(final_base_text)



"----------------------------------------------------------------------------"
#%%

"Named Entity Recognition (NER)"

import spacy
nlp = spacy.load("en_core_web_sm")
nlp.pipe_names #pipe_names are processing components in the nlp pipeline

"""
tok2vec: Creates vector representations of words and subwords, capturing contextual information.

tagger: Assigns part-of-speech tags (e.g., noun, verb, adjective) to each word.

parser: Analyzes the grammatical structure of the sentence, identifying relationships between words (dependency parsing).

ner: Named Entity Recognition. Identifies and classifies named entities like people, organizations, locations, etc.

attribute_ruler: A component for adding custom rules or exceptions to the pipeline, often used for handling specific linguistic patterns.

lemmatizer: Reduces words to their base or dictionary form (lemma). For example, "running" becomes "run".

"""


doc = nlp("Tesla Inc is going to acquire twitter for $45 billion")
for ent in doc.ents: #doc.ents: entities in the documents
    print(ent.text, " | ", ent.label_, " | ", spacy.explain(ent.label_))

#ent.text: The actual text of the named entity (e.g., "Tesla Inc").
#ent.label_: The entity label assigned by the NER component (e.g., "ORG" for organization).
#spacy.explain(ent.label_): This uses the spacy.explain() function to provide a human-readable description of the entity label (e.g., "Companies, agencies, institutions, etc." for "ORG").

"""
run this in jupyter for better looking output version

from spacy import displacy

displacy.render(doc, style="ent")
"""

#listing all the entities 

nlp.pipe_labels['ner']

"""
CARDINAL: Cardinal numbers (e.g., one, two, 10, 100).

DATE: Absolute or relative dates or periods (e.g., yesterday, today, 2020-09-01, next week).

EVENT: Named hurricanes, battles, wars, sports events, etc. (e.g., Hurricane Katrina, World War II).

FAC: Buildings, airports, highways, bridges, etc. (e.g., Empire State Building, JFK Airport).

GPE: Countries, cities, states (e.g., United States, London, California).

LANGUAGE: Any named language (e.g., English, Spanish, Python).

LAW: Named documents made into laws. (e.g., Patriot Act, First Amendment).

LOC: Non-GPE locations, mountain ranges, bodies of water (e.g., Mount Everest, Amazon River).

MONEY: Monetary values, including unit (e.g., $50, 100 euros).

NORP: Nationalities or religious or political groups (e.g., American, Christian, Republican).

ORDINAL: Ordinal numbers (e.g., first, second, 1st, 2nd).

ORG: Companies, agencies, institutions, etc. (e.g., Apple, FBI, United Nations).

PERCENT: Percentage, including "%" (e.g., 90%, 3.5%).

PERSON: People, including fictional (e.g., Barack Obama, Harry Potter).

PRODUCT: Objects, vehicles, foods, etc. (e.g., iPhone, Tesla Model S, Big Mac).

QUANTITY: Measurements, as of weight or distance (e.g., 10 kg, 5 miles).

TIME: Times smaller than a day (e.g., 2:30pm, 11:00).

WORK_OF_ART: Titles of books, songs, etc. (e.g., The Lord of the Rings, Bohemian Rhapsody).

# List of entities are also documented on this page: https://spacy.io/models/en

"""


doc = nlp("Michael Bloomberg founded Bloomberg in 1982")
for ent in doc.ents:
    print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))

# in the otuput, there seems error : bloomberg is classified as person
# it is an ORG

# you can try it here : https://huggingface.co/dslim/bert-base-NER?text=Michael+Bloomberg+founded+Bloomberg+in+1982


doc = nlp("Tesla Inc is going to acquire Twitter Inc for $45 billion")
for ent in doc.ents:
    print(ent.text, " | ", ent.label_, " | ", ent.start_char, "|", ent.end_char)


#%%

"""
TF-IDF (Term Frequency - Inverse Document Frequency)

TERM FREQUENCY 
TF stands for Term Frequency and denotes the ratio of number of times a particular word appeared in a Document to total number of words in the document.

Term Frequency(TF) = [number of times word appeared / total no of words in a document]
Term Frequency values ranges between 0 and 1.
If a word occurs more number of times, then it's value will be close to 1.


IDF, or Inverse Document Frequency, measures how rare a word is across a set of documents. 
A high IDF means a word is uncommon and likely more informative, 
while a low IDF means a word is frequent and probably less informative.

Think of it like this: words like "the" or "is" appear in almost every document, 
so they don't tell you much about the specific content of a particular document. But a word like "algorithm" or "quantum" is much rarer, so if it appears in a document, it's a strong indicator of what the document is about. IDF captures this distinction.

DF stands for Inverse Document Frequency and denotes the
log of ratio of total number of documents/datapoints in the whole dataset to the number of documents that contains the particular word.

Inverse Document Frequency(IDF) = 
[log(Total number of documents / number of documents that contains the word)]

In IDF, if a word occured in more number of documents and is common across all documents, then it's value will be less and ratio will approaches to 0.

Finally:

   TF-IDF = Term Frequency(TF) * Inverse Document Frequency(IDF)


"""
#%%

"basic example to start with......"

#What is the TF of each word in the document: "The quick brown fox jumps over the lazy dog"?

from collections import Counter
# Counter is a useful tool for counting the frequency of items in a list.

def calculate_tf(document):
    """Calculates Term Frequency (TF) for each word in a document."""
    word_counts = Counter(document)
    print("counts:",word_counts)
    "The Counter takes the document (a list of words) and counts how many times \
        each word appears. It stores these counts like a dictionary (word: count)."
    total_words = len(document)
    print("length:",total_words)#takes length of all words in the document and stored in total_words
    tf_scores = {} 
    for word, count in word_counts.items(): #.items() display key value pair
        tf_scores[word] = count / total_words
    return tf_scores

document = "The quick brown fox jumps over the lazy dog".split()  # Split into words
tf = calculate_tf(document)

for word, score in tf.items():
    print(f"TF({word}) = {score:.4f}")

"""
Observations: 
    
each word appears once in a document of nine words, 
so its TF is 1/9 ≈ 0.125. This simple scenario provides a clear and concise illustration of the core concept of TF.

"""





#%%

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already",
    "Apple is announcing new iphone tomorrow",
    "Tesla is announcing new model-3 tomorrow",
    "Google is announcing new pixel-6 tomorrow",
    "Microsoft is announcing new surface tomorrow",
    "Amazon is announcing new eco-dot tomorrow",
    "I am eating biryani and you are eating grapes"
]

#let's create the vectorizer and fit the corpus and transform them accordingly
v = TfidfVectorizer()
v.fit(corpus)
transform_output = v.transform(corpus)

#let's print the vocabulary

print(v.vocabulary_)

#let's print the idf of each word:

all_feature_names = v.get_feature_names_out()

for word in all_feature_names:
    
    #let's get the index in the vocabulary
    indx = v.vocabulary_.get(word)
    
    #get the score
    idf_score = v.idf_[indx]
    
    print(f"{word} : {idf_score}")
    
    
    
    
    
    
    
    

#%%

"""
BAG OF WORDS

Imagine you have a bag of words, literally. You reach in and pull out words one by one, but you don't care about the order in which you pull them out. You just count how many times each word appears. That's essentially what the "Bag of Words" model does in NLP.

It represents text as a collection (or "bag") of individual words, disregarding grammar and word order but keeping track of word frequency.

Example:

Sentence 1: "The quick brown fox jumps over the lazy dog."
Sentence 2: "The dog barked at the quick fox."

Bag of Words representation:

Word	Sentence 1	Sentence 2
the	2	2
quick	1	        1
brown	1	        0
fox	1	1
jumps	1	        0
over	1	        0
lazy	1	        0
dog	1	1
barked	0	        1
at	    0	        1

Now zoom out, and think for heavier corpus : books, articles, etc

Limitations  : 

Vocabulary(article,book) is long, if 100k tokens, then each vector 
for each article will be around 100k
better than one hot encoding, because it will add lot more dimensions

It is mostly sparse presentation (most of the elements in vector are 0)
consumes lot of memory
does not capture the meaning, we only take word count

numeric representation , that's all

"""
from collections import Counter
import pandas as pd

def bag_of_words(documents):
    """Creates a Bag-of-Words representation of a list of documents."""

    tokenized_documents = [doc.lower().split() for doc in documents]

    # Vocabulary Building (without set)
    vocabulary = []
    for doc in tokenized_documents:
        for word in doc:
            if word not in vocabulary:  # Check for duplicates
                vocabulary.append(word)
    vocabulary.sort() # Sort the vocabulary


    # Create Bag-of-Words Representation (same as before)
    bow_representation = []
    for doc in tokenized_documents:
        word_counts = Counter(doc)
        bow_vector = [word_counts.get(word, 0) for word in vocabulary]
        bow_representation.append(bow_vector)

    return pd.DataFrame(bow_representation, columns=vocabulary)

# Example Usage (same as before):
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog barked at the quick fox.",
    "A brown cat sat on the mat."
]


bow_df = bag_of_words(documents)
print(bow_df)




#%%

