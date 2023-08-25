
from sentence_transformers import SentenceTransformer, util
from basic_cleaner import clean
import requests
import json
import spacy
import string

from summa import summarizer
import time


model_base = SentenceTransformer('thenlper/gte-base', device='cuda')
model_large = SentenceTransformer('thenlper/gte-large', device='cuda')
model_bge_large = SentenceTransformer('BAAI/bge-large-en', device='cuda')
model_e5_large_v2 = SentenceTransformer('efederici/e5-large-v2-4096', {"trust_remote_code": True})

model_e5_large_v2.max_seq_length= 4096


def str_2_list_of_str(s):
    """
    Convert a string to a list of strings.
    """
    s= s.replace('[', '')
    s= s.replace(']', '')
    s= s.replace('\n', '')
    s= s.replace('\t', '')
    s= s.replace('  ', '')
    s= s.replace('"', '')
    s= s.replace("'", '')
    list_of_strings= s.split(',')
    return list_of_strings


def generate_base_embeddings(text): 
    """
    Generate embeddings for the given text using GTE-base.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    embeddings= model_base.encode(text, convert_to_tensor=True)
    
    
    # return util.cos_sim(embeddings[0], embeddings[1])
    return embeddings.cpu().numpy()




def generate_e5_large_v2_embeddings(text): 
    """
    Generate embeddings for the given text using e5_large_v2.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    embeddings= model_e5_large_v2.encode(text, convert_to_tensor=True)
    
    
    # return util.cos_sim(embeddings[0], embeddings[1])
    return embeddings.cpu().numpy()








def generate_large_embeddings(text):
    """
    Generate embeddings for the given text using GTE-large.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    
    embeddings= model_large.encode(text, convert_to_tensor=True)
    # return util.cos_sim(embeddings[0], embeddings[1])
    return embeddings.cpu().numpy()


def generate_bge_large_embeddings(text):
    """
    Generate embeddings for the given text using BGE-large.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    
    embeddings= model_bge_large.encode(text, convert_to_tensor=True)
    # return util.cos_sim(embeddings[0], embeddings[1])
    return embeddings.cpu().numpy()






def generate_cosine_similarity(e1, e2):
    """
    Generate cosine similarity for the given embeddings.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    
    # embeddings= model_bge_large.encode(text, convert_to_tensor=True)
    # # return util.cos_sim(embeddings[0], embeddings[1])
    # return embeddings.cpu().numpy()
    return util.cos_sim(e1, e2)
   
   
   


def generate_keyword_summary(keyword):
    """Generate a summary of the keyword"""
    response= requests.api.get(f'https://2qq35q1je7.execute-api.us-east-1.amazonaws.com/?search={keyword}')
    d= json.loads(response.text)
    
    data= d['data']
    results= data['results']
    
    s= ""
    

    for i in results:
        s+=i['url']+' '
        s+=i['description']
        
    s= s.replace("https://", '')
    s= s.replace("/", '')
    s= s.replace(",", '')
    s= s.replace("www.", '')
        
    summary=summarizer.summarize(s, words=200).replace('\n', ' ')
    # summary= spacy_tokenizer(s)
    
    return summary 
    




punctuations = string.punctuation
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)
    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    sentence = " ".join(mytokens)
    # return preprocessed list of tokens
    return sentence
