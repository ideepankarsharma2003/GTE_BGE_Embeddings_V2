from http.client import HTTPSConnection
from base64 import b64encode
from json import loads
from json import dumps

class RestClient:
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def request(self, path, method, data=None):
        connection = HTTPSConnection(self.domain)
        try:
            base64_bytes = b64encode(
                ("%s:%s" % (self.username, self.password)).encode("ascii")
                ).decode("ascii")
            headers = {'Authorization' : 'Basic %s' %  base64_bytes, 'Content-Encoding' : 'gzip'}
            connection.request(method, path, headers=headers, body=data)
            response = connection.getresponse()
            return loads(response.read().decode())
        finally:
            connection.close()

    def get(self, path):
        return self.request(path, 'GET')

    def post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = dumps(data)
        return self.request(path, 'POST', data_str)



client = RestClient("kumar@warewe.com", "fd5a40e08700c882")
# client = RestClient("deepankar@warewe.com", "cb1661e9ec7c1fba")


def generate_seo_metatitle(keyword):
    post_data = dict()
    # You can set only one task at a time
    post_data[len(post_data)] = dict(
        language_code="en",
        location_code=2840,
        keyword=keyword
    )
    
    response = client.post("/v3/serp/google/organic/live/regular", post_data)



    if response["status_code"] == 20000:
        # print(response)
        d= response['tasks'][0]
        # print(d)
        result_dict= d['result'][0]['items']
        summary= ''
        for i in result_dict[:25]:
            summary+= i['title']+' '

        return summary
        
        
        # do something with result
    else:
        print("error. Code: %d Message: %s" % (response["status_code"], response["status_message"] , f" for keyword {keyword}"))