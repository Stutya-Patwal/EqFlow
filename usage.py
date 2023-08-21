#import the module
from EqFlow import Eqre as eqre
from EqFlow import Eqai as eqai

#________________________________________________________________
# FOR BS4 AND REQUESTS MODULE: EQRE
#_______________________________________________________________
#store the url in a var
url = "https://www.google.com/search?q=equilateral+triangle"

#verify the url
#check if the url is working
eqre.verify_url(url)

#getting the soup
#get the content of the page
soup = eqre.get_soup(url)

#get the text present in the page (raw)
#this is raw
text = eqre.get_raw_text(soup)

#get the text present in the page (processed using ai)
#this is processed using ai (better than raw, removes uncessecary things) (beta)
ptext = eqre.get_processed_text(soup)

#find elements
felement = eqre.find(url,id="yKMVIe")#class_,text,tag or id


#________________________________________________________________
# AI STUFF: EQAI
#________________________________________________________________
# Auto-stored input text
input_text = "This is an example sentence." 

# Tokenization
tokens = eqai.tokenize_text(input_text)

# Part-of-Speech Tagging
pos_tags = eqai.pos_tagging(input_text)

# Named Entity Recognition
named_entities = eqai.named_entity_recognition(input_text)

# Sentiment Analysis
sentiment = eqai.sentiment_analysis(input_text)

# Text Summarization
summary = eqai.text_summarization(input_text)
