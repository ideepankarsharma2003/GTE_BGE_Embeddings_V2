import uvicorn
import sys
import os
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from main_transformers import generate_base_embeddings, generate_large_embeddings, str_2_list_of_str






app= FastAPI()
print("initializing app")

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')
    # return "Hello world!"


 



@app.get('/base')
async def base(text):
    
    try: 
        text= str_2_list_of_str(text)
        
        
        # print(type(text))
        # print(text)
        print(f"n_urls: {len(text)}")
        
        
        embeddings= generate_base_embeddings(text)
        # print(embeddings.shape)
        print(f"embeddings: {embeddings.shape}")
        # print(embeddings)
        return embeddings.tolist()
        # return (embeddings[0][0].item())
    except Exception as e:
        return Response(f'Error occured: {e}')



@app.get('/large')
async def base(text):
    
    try: 
        text= str_2_list_of_str(text)
        embeddings= generate_large_embeddings(text)
        print(f"n_urls: {len(text)}")
        print(f"embeddings: {embeddings.shape}")

        # return (embeddings[0][0].item())
        return embeddings.tolist()
    except Exception as e:
        return Response(f'Error occured: {e}')



if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8081)