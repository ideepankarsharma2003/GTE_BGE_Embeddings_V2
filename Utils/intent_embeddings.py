import numpy as np
import json
import pickle

intents= {
    'informational': '''Informational intent is when users are seeking information, answers, or explanations. They want to learn about a specific topic, understand a concept, or find facts. These searches often begin with question words like "what," "how," "why," or "who." Example: "What is photosynthesis?" Goal: The user wants to understand the process of photosynthesis and gain knowledge about it.''',
    
    'navigational':'''Informational intent is when users are seeking information, answers, or explanations. They want to learn about a specific topic, understand a concept, or find facts. These searches often begin with question words like "what," "how," "why," or "who." Example: "What is photosynthesis?" Goal: The user wants to understand the process of photosynthesis and gain knowledge about it.''',
    
    'transactional': '''Transactional intent is driven by users who intend to perform a specific action, such as making a purchase, signing up for a service, or downloading something.Example: "Buy iPhone 12 online"Goal: The user intends to make a purchase of an iPhone 12 online.''',
    
    'commercial': '''Commercial investigational intent is when users are in the research phase of a potential purchase. They are comparing products, reading reviews, and seeking information before making a decision.Example: "Best laptop for graphic design"Goal: The user is looking for information about laptops suitable for graphic design work, with the intention of purchasing one.''',
    
    'local': '''Local intent is when users are searching for products, services, or information specific to a particular location. These searches often include phrases like city names or "near me." Example: "Restaurants near me" Goal: The user wants to find nearby restaurants for dining out.'''
}







# with open('Utils/embeddings.json', 'r') as f:
#     s= f.read()    
# data = json.loads(s)
# intent_embeddings= np.array(data, dtype=np.double)

intent_embeddings= pickle.load(open('Utils/intent_embeddings.pkl', 'rb'))









reverse_intent= {
    0: 'informational',
    1: 'navigational',
    2: 'transactional',
    3: 'commercial',
    4: 'local'
}
