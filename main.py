
from sentence_transformers import SentenceTransformer, util
from basic_cleaner import clean

model_base = SentenceTransformer('thenlper/gte-base', device='cuda')
model_large = SentenceTransformer('thenlper/gte-large', device='cuda')



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
    
    


