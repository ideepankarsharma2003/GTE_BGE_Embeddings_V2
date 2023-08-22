import requests
from bs4 import BeautifulSoup
from summa import summarizer



def clean(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unnecessary elements (styles, scripts, etc.)
        for script in soup(['script', 'style', 'a', 'img', 'video']):
            script.extract()
        
        cleaned_text = soup.get_text()
        cleaned_text= cleaned_text.strip()
        cleaned_text= cleaned_text.replace('\n', ' ')
        cleaned_text= cleaned_text.replace('\t', ' ')
        
        return summarizer.summarize(cleaned_text, words=150)
            
    else:
        return ("Failed to retrieve the webpage")

