#streamlit
from http.client import CannotSendHeader
from re import template
from sre_constants import SRE_FLAG_LOCALE
import streamlit as st
import pandas as pd


#stuff
import json
import time
import os

#classification
from transformers import pipeline

#speech recognition
import speech_recognition as sr

#import pocketsphinx


# #chatbot
import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from urllib import request, response
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot import comparisons
from chatterbot import response_selection

#text to speech
import pyttsx3
text_speech = pyttsx3.init()

#sentiment analysis
from textblob import TextBlob


########################## FUNCTION    ############
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0.05:
        sentiment_label = 'Positive'
    elif sentiment_polarity < -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    return result
####################### initializations #############

# voices = text_speech.getProperty('voices')
# text_speech.setProperty('voice', voices[37].id)

category_template = {
    'FnB waiter': ['show menu', 'order food', 'food description', 'assistance request', 'billing', 'social interaction','who are you'],
    'delivery robot': ['make a delivery', 'billing','task enquiry', 'social interaction','who are you'],
    'hospital nursing robot' : ['assistance request', 'negative physical health', 'social interaction', 'social need', 'who are you'],
    'social robot' : ['social interaction', 'social need', 'trivia questions'],
    'customize your own' : []
}


if 'chatbot' not in st.session_state:
        
    conversation_corpus = open('conversation_corpus.txt', 'r')

    training_ls = []
    while True:

        # Get next line from file
        line = str(conversation_corpus.readline())

        if not line:
            break

        line = line.replace('Human 1: ', '')
        line = line.replace('Human 2: ', '')
        line = line.replace('\n', '')
        training_ls.append(line)


    conversation_corpus.close()
    #chatbot
    chatbot = ChatBot(
        'Florence',
        #storage_adapter='chatterbot.storage.SQLStorageAdapter',
        logic_adapters=[
            {
                "import_path": "chatterbot.logic.BestMatch",
                "statement_comparison_function": chatterbot.comparisons.LevenshteinDistance,
                "response_selection_method": chatterbot.response_selection.get_first_response
            },
            'chatterbot.logic.MathematicalEvaluation', 
            #'chatterbot.logic.TimeLogicAdapter',

        ]
        #database_uri='sqlite:///database.db'
    )


    trainer = ListTrainer(chatbot)
    trainer.train(training_ls)

    st.session_state['chatbot'] = chatbot


#######################################
####################### UI ####################
header = st.container()
choose_template = st.container()
voice_container = st.container()

#about
with header:
    st.title('NLP For Robots Demo')
    st.write('DISCLAIMER: This is just a demo, the model is not trained with the real company data.')
    st.title("Create your own speech NLP engine!")
    st.write('contextualize speech and analyze their sentiment to provide appropriate response.')
    st.image('nlp_flowchart.png')


    st.write('_____________________________')
    st.write('')

#step 1
with choose_template:
    st.header('STEP 1: load your model')
    sel_col , dis_col = st.columns(2,gap = "large")
    sel_col.header('what would be the purpose of your robot?')
    robot_type = sel_col.selectbox('choose a template or define your own:', options = ['select an option', 'FnB waiter', 'delivery robot', 'hospital nursing robot', 'social robot', 'customize your own'])
    
    dis_col.header('your conversational categories:')
    if robot_type == 'select an option':
        dis_col.write('your categories will appear here') 
        next_button = ''
    elif robot_type == 'customize your own':

        text_input = dis_col.text_input('type in your categories seperated with comas')
        category_template[robot_type] = text_input.split(', ')
        dis_col.write(category_template[robot_type])
        next_button = dis_col.button('click here to load your model')
    else:
        dis_col.write(category_template[robot_type])
        next_button = dis_col.button('click here to load your model')

# with voice_container:

#next button
if next_button and len(category_template[robot_type]) < 2:
    st.warning('you must input at least 2 categories')

elif next_button:
    text = st.warning('loading model... (this may take a few seconds)')
    st.session_state['candidates'] = category_template[robot_type]
    if 'classifier' not in st.session_state:
        st.session_state['classifier'] = pipeline("zero-shot-classification")
    text = st.warning('loading completed')



        ########################## temporary ########
        # r = sr.Recognizer()

        # with sr.Microphone() as source:
        #     st.subheader("speak to the microphone: ")
        #     audio = r.listen(source)
        #     try:
        #         text = r.recognize_google(audio)
        #         st.write('you: ' + text) #what you said
                
        #         #categorical analysis

        #         category = classifier(text, candidate_labels=candidates)
        #         sentiment = get_sentiment(text)['sentiment']
        #         st.text("type of interaction: "+ category['labels'][0] +', sentiment:' + sentiment)
        #     except:
        #         st.write('Sorry, I cant hear you properly')
        ###############################



st.write('_____________________________')
st.write('')

#step 2
st.header('STEP 2: test your model')

if 'candidates' not in st.session_state:
    st.text('No model is loaded yet. please choose a model and click on the "load model" button in step 1')
else:
    st.text('classifyling:')
    st.session_state['candidates']




sel_col2 , dis_col2 = st.columns(2,gap = "large")
sel_col2.subheader('select testing type:')
test_type = sel_col2.selectbox('  ', options = ['select an option', 'test with speech input','test with keyboard input'])

dis_col2.subheader('your results:')
if test_type == 'select an option':
    if st.session_state == {}:
        st.warning('please load your model first!')
    else:
        dis_col2.text("category: ")
        dis_col2.text('sentiment:')


elif test_type == 'test with speech input':
        if st.session_state == {}:
            st.warning('please load your model first!')
        else:

            next_button = dis_col2.button('press to start')

            if next_button:
                time.sleep(5)
                r = sr.Recognizer()

                with sr.Microphone() as source:

                    dis_col2.text("speak to the microphone: ")
                    audio = r.listen(source)

                    try:
                        text = r.recognize_google(audio)
                        dis_col2.write('you: ' + text) #what you said
                        
                        #categorical analysis

                        category = st.session_state['classifier'](text, candidate_labels=st.session_state['candidates'])
                        sentiment = get_sentiment(text)['sentiment']
                        dis_col2.text("category: "+ category['labels'][0])
                        dis_col2.text('sentiment:' + sentiment)
                    except:
                        dis_col2.write('Sorry, I cant hear you properly')

# elif test_type == 'test with continuous speech':
#     if st.session_state == {}:
#         st.warning('please load your model first!')
#     else:
#         while True:
#             r = sr.Recognizer()

#             with sr.Microphone() as source:
#                 dis_col2.text("speak to the microphone: ")
#                 audio = r.listen(source)
#                 try:
#                     text = r.recognize_google(audio)
#                     dis_col2.write('you: ' + text) #what you said
                    
#                     #categorical analysis

#                     category = st.session_state['classifier'](text, candidate_labels=st.session_state['candidates'])
#                     sentiment = get_sentiment(text)['sentiment']
#                     dis_col2.text("category: "+ category['labels'][0])
#                     dis_col2.text('sentiment:' + sentiment)
#                 except:
#                     dis_col2.write('Sorry, I cant hear you properly')     

elif test_type == 'test with keyboard input':
    if st.session_state == {}:
        st.warning('please load your model first!')
    else:

        text = dis_col2.text_input('type something')
        if text =='':
            dis_col2.text("category: ")
            dis_col2.text('sentiment: ')
        else:
            result = st.session_state['classifier'](text, candidate_labels = st.session_state['candidates'])
            category = result['labels'][0]

            #dis_col2.text("you: " + text) #what you said
            dis_col2.text("category: "+ category)
            dis_col2.text('sentiment: ' + get_sentiment(text)['sentiment'])

    


st.write('_____________________________')
st.write('')
st.header('STEP 3: customize a response (DEMO)')
st.write('a demo on how the output can be used to generate proper responses for a nursing robot')
st.image('florence_flowchart.png')


florence_candidates = category_template['hospital nursing robot']




st.subheader('interact with the nursing robot')

sel_col3, dis_col3 = st.columns(2, gap='large')
sel_col3.caption('')
demo_select = sel_col3.selectbox('choose your type of interaction', ['speech', 'keyboard'])

dis_col3.write('output:')

if demo_select == 'speech':
    demo_button = dis_col3.button('start')

    if demo_button:
        r = sr.Recognizer()
        loop = True
        widget_id = 0

        while loop:
            loop = False 

            if text_speech._inLoop:
                text_speech.endLoop()

            with sr.Microphone() as source:
                dis_col3.text("speak to the microphone: ")
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)

                except:
                    dis_col3.write('Sorry, I cant hear you properly')     

            widget_id += 1

            if 'classifier' not in st.session_state:
                st.warning('loading model... (this may take a few seconds)')
                st.session_state['classifier'] = pipeline("zero-shot-classification")
                
            elif text == '':
                st.warning('please type in an input.')
            else:
                result = st.session_state['classifier'](text, candidate_labels=florence_candidates)
                category = result['labels'][0]

                dis_col3.text("you: " + text + ' ({})'.format(category)) #what you said


                if category == "social interaction" or category == "others" or category == "social need" or category == "object":


                    if text_speech._inLoop:
                        text_speech.endLoop()

                    if text == 'bye':
                        dis_col3.text("Florence: bye \n")
                        text_speech.say('bye!')
                        text_speech.runAndWait()
                        break

                    else:
                        response = st.session_state['chatbot'].get_response(text)
                        dis_col3.text('Florence: ' + str(response))
                        text_speech.say(str(response))
                        text_speech.runAndWait()
                        loop = True
                        
                elif category == "assistance request":

                    if text_speech._inLoop:
                        text_speech.endLoop()

                    dis_col3.text('Florence: Okay, Florence will help you!')
                    text_speech.say('Okay, Florence will help you!')
                    text_speech.runAndWait()
                    
                elif category == "who are you":

                    if text_speech._inLoop:
                        text_speech.endLoop()

                    dis_col3.text("Florence: I am Florence, your AI nursing robot")
                    text_speech.say("I am Florence, your AI nursing robot")
                    text_speech.runAndWait()                    
                    loop = True


                else:
                    loop2 = True

                    if text_speech._inLoop:
                        text_speech.endLoop()            
                    while loop2:

                        loop2 = False
                        dis_col3.text("Florence: rate how bad is the situation from 1 to 10")
                        text_speech.say("rate how bad is the situation from 1 to 10")
                        text_speech.runAndWait()



                        with sr.Microphone() as source:
                            dis_col3.text("speak to the microphone: ")
                            audio = r.listen(source)
                            try:
                                rating = r.recognize_google(audio)
                                dis_col3.write('you: ' + text) #what you said
                                

                            except:
                                dis_col3.write('Sorry, I cant hear you properly')     



                        not_call_nurse = ["1", "2", "3", "one", "two", "three"]
                        call_nurse = ['4', '5', '6', '7', '8', '9', '10', "four", 'five', 'six', 'seven', 'eight', 'nine', 'ten']
                        
                        
                        if rating == '':
                            st.warning('please input rating')
                        elif rating in not_call_nurse:

                            if text_speech._inLoop:
                                text_speech.endLoop()

                            response = st.session_state['chatbot'].get_response(text)
                            dis_col3.text('Florence: ' + str(response))
                            text_speech.say(str(response))
                            text_speech.runAndWait()     
                            loop = True
                            break

                        elif rating in call_nurse:


                            if text_speech._inLoop:
                                text_speech.endLoop()

                            dis_col3.text("Florence: let me get a nurse")
                            text_speech.say('let me get a nurse')
                            text_speech.runAndWait()                            
                            
                            break

                        else:


                            if text_speech._inLoop:
                                text_speech.endLoop()

                            dis_col3.text("Florence: Sorry, please say again")
                            dis_col3.text("Sorry, please say again")
                            text_speech.say('Sorry, please say again')
                            loop2 = True



  

elif demo_select =='keyboard':
    loop = True
    widget_id = 0
    while loop:
        loop = False 

        text = dis_col3.text_input('type to interact:', key = widget_id)
        widget_id += 1
        if 'classifier' not in st.session_state:
            st.warning('loading model... (this may take a few seconds)')
            st.session_state['classifier'] = pipeline("zero-shot-classification")
            

        elif text == '':
            st.warning('please type in an input.')
        else:
            result = st.session_state['classifier'](text, candidate_labels=florence_candidates)
            category = result['labels'][0]

            dis_col3.text("you: " + text + ' ({})'.format(category)) #what you said


            if category == "social interaction" or category == "others" or category == "social need" or category == "object":

                if text == 'bye':
                    dis_col3.text("Florence: bye \n")
                    break

                else:
                    response = st.session_state['chatbot'].get_response(text)
                    dis_col3.text('Florence: ' + str(response))
                    loop = True
                    
            elif category == "assistance request":
                dis_col3.text('Florence: Okay, Florence will help you!')
                
            elif category == "who are you":
                dis_col3.text("Florence: I am Florence, your AI nursing robot")
                loop = True


            else:
                loop2 = True
                while loop2:
                    loop2 = False

                    dis_col3.text("Florence: rate how bad is the situation from 1 to 10")
                            
                    rating = dis_col3.text_input("rate: ", key = widget_id)

                    not_call_nurse = ["1", "2", "3", "one", "two", "three"]
                    call_nurse = ['4', '5', '6', '7', '8', '9', '10', "four", 'five', 'six', 'seven', 'eight', 'nine', 'ten']
                    
                    
                    if rating == '':
                        st.warning('please input rating')
                    elif rating in not_call_nurse:
                        response = st.session_state['chatbot'].get_response(text)
                        dis_col3.text('Florence: ' + str(response))
                        loop = True
                        break

                    elif rating in call_nurse:
                        dis_col3.text("Florence: let me get a nurse")
                        break

                    else:
                        dis_col3.text("Florence: Sorry, please say again")
                        loop2 = True

