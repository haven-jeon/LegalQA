import base64
from collections import OrderedDict
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

headers = {
    'Content-Type': 'text/text; charset=utf-8'
}

# Set default endpoint. Usually this would be passed to a function via a parameter
DEFAULT_ENDPOINT = "http://0.0.0.0:45678/api/search"


class text:
    """
    Jina text search
    """

    class process:
        """
        Process query and results
        """

        def json(query: str, top_k: int, endpoint: str) -> list:
            """
            Process Jina's JSON output and parse results
            """
            data = {"top_k": top_k, "mode": "search", "data": [query]}
            response = requests.post(endpoint, headers=headers, json=data)

            content = response.json()["search"]["docs"]
            results = []
            for doc in content:
                matches = doc["matches"]  # list
                for match in matches:
                    score = match["score"]["value"]
                    title = match["text"]
                    body = match["tags"]["answer"]
                    results.append(OrderedDict({'score': score, 'title': title, 'body': body}))

            return results


class image:
    """
    Jina image search
    """

    class encode:
        """
        Encode image to base64 and return JSON string
        """

        def img_base64(byte_string):
            """
            Encode image file to base64
            """
            output = str(base64.b64encode(byte_string))[2:-1]
            output = f'["data:image/png;base64,{output}"]'

            return output

    class process:
        def json(query: str, top_k: int, endpoint: str) -> list:
            data = (
                '{"top_k":' + str(top_k) + ', "mode": "search", "data":' + query + "}"
            )
            response = requests.post(endpoint, headers=headers, data=data)

            content = response.json()["search"]["docs"]
            results = []
            for doc in content:
                matches = doc["matches"]
                for match in matches:
                    results.append(match["uri"])

            return results

    class render:
        """
        Render image output
        """

        def html(results: list) -> str:
            """
            Render images as list of HTML img tags
            """
            output = ""
            for doc in results:
                html = f'<img src="{doc}">'
                output += html

            return output


class jina:
    def text_search(endpoint=DEFAULT_ENDPOINT, top_k=10, hidden=[]):
        container = st.beta_container()
        with container:
            if "endpoint" not in hidden:
                endpoint = st.text_input("Endpoint", endpoint)

            query = st.text_input("Enter query", value='조의금 소유는 누가 하는건가요?')
            print(f'Query: {query}')

            if "top_k" not in hidden:
                top_k = st.slider("Results", 1, top_k, int(top_k / 2))

            button = st.button("Search")

            if button:
                matches = text.process.json(query, top_k, endpoint)
                st.write(matches)
                print(matches)
                print()

        return container

    def image_search(endpoint=DEFAULT_ENDPOINT, top_k=10, hidden=[]):
        container = st.beta_container()
        with container:
            if "endpoint" not in hidden:
                endpoint = st.text_input("Endpoint", endpoint)

            query = st.file_uploader("Upload file")

            if "top_k" not in hidden:
                top_k = st.slider("Results", 1, top_k, int(top_k / 2))

            button = st.button("Search")

            if button:
                # encode to base64 and embed in json
                encoded_query = image.encode.img_base64(query.read())
                # post to REST API and process response
                matches = image.process.json(encoded_query, top_k, endpoint)
                # convert list of matches to html
                output = image.render.html(matches)
                # render html
                return st.markdown(output, unsafe_allow_html=True)

        return container


st.set_page_config(page_title="Jina Text Search",)

endpoint = "http://0.0.0.0:1234/api/search"

st.title("LegalQA with SentenceKoBART")
st.markdown("")

jina.text_search(endpoint=endpoint, hidden=['endpoint'])
