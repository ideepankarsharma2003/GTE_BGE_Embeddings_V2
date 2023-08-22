
from sentence_transformers import SentenceTransformer, util
from basic_cleaner import clean
import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


model_base = AutoModel.from_pretrained("thenlper/gte-base",
    # device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True)

model_large = AutoModel.from_pretrained("thenlper/gte-large",
    # device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True)

tokenizer_base = AutoTokenizer.from_pretrained("thenlper/gte-base")
tokenizer_large = AutoTokenizer.from_pretrained("thenlper/gte-large")
# model_large = SentenceTransformer('thenlper/gte-large', device='cuda')



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
    batch_dict = tokenizer_base(text, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model_base(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings






def generate_large_embeddings(text):
    """
    Generate embeddings for the given text using GTE-large.
    """
    # for i in range(len(text)):
    #     text[i]= clean(text[i])
    #     print(text[i])
    # print()
    
    batch_dict = tokenizer_large(text, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model_large(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings

    
    


