import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()



data = pd.read_csv('Mental_Health_FAQ.csv')

data.drop('Question_ID' , axis = 1, inplace = True)



# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
       
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)

data['tokenized Questions'] = data['Questions'].apply(preprocess_text)

x = data['tokenized Questions'].to_list()

tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(x)




st.markdown("<h1 style = 'color: #5E1675; text-align: center; font-family: helvetica'>MENTAL HEALTH CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #5F374B; text-align: center; font-family: cursive '>Built by Ilo M.A</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

hist_list = []

robot_image, chat_response = st.columns(2)
with robot_image:
    robot_image.image('pngwing.com (3).png', caption = 'I reply all your questions')
    

with chat_response :
      user_word = chat_response.text_input('Hello there you can ask your questions: ')  
def get_response(user_input):
    user_input_processed = preprocess_text(user_input)

    user_input_vector = tfidf_vectorizer.transform([user_input_processed])

    similarity_scores = cosine_similarity(user_input_vector, corpus) 

    most_similar_index = similarity_scores.argmax() 
    return data['Answers'].iloc[most_similar_index] 

# create greeting list 
greetings = ["Hey There.... I am a creation of Motun Agba Coder.... How can I help",
            "Hi Human.... How can I help",
            'Twale baba nla, wetin dey happen nah',
            'How far Alaye, wetin happen'
            "Good Day .... How can I help", 
            "Hello There... How can I be useful to you today",
            "Hi GomyCode Student.... How can I be of use"]

exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
farewell = ['Thanks....see you soon', 'Babye, See you soon', 'Bye... See you later', 'Bye... come back soon']

random_farewell = random.choice(farewell) 
random_greetings = random.choice(greetings) 

# Test your chatbot
# while True :
#       user_input = input("You: ")
if user_word.lower() in exits:
            chat_response.write(f"\nChatbot: {random_farewell}!")

elif user_word.lower() in ['hi', 'hello', 'hey', 'hi there']:
         chat_response.write(f"\nChatbot: {random_greetings}!")

elif user_word == '':
         chat_response.write('')
        
else:   
        response = get_response(user_word)
        chat_response.write(f"\nChatbot: {response}")

        hist_list.append(user_word)

#save the history of the texts
with open('history.txt', 'a') as file:
        for item in hist_list:
            file.write(str(item) + '\n')
            file.write(response)    

        import csv
        files = 'history.txt' 
        with open(files) as f:
             reader = csv.reader(f)
             data = list(reader)
          



history = pd.Series(data)
st.sidebar.subheader('Chat History', divider = True)
st.sidebar.write(history)


st.header('Project Background Information',divider = True)
st.write("In response to the increasing prevalence of mental health challenges, we have developed a compassionate chatbot named Mind Matters. This chatbot aims to provide accessible and personalized support for individuals navigating their mental well-being, offering a safe space to express feelings, access resources and receive guidance")
