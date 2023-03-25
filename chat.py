import streamlit as st
from streamlit_chat import message
import requests

st.set_page_config(
    page_title="LegalQA - Demo",
    page_icon=":robot:"
)

API_URL = "http://localhost:1234/search"
headers = {
    'Content-Type': 'application/json; charset=utf-8'
}

st.header("LegalQA - Demo")
st.markdown("[https://github.com/haven-jeon/LegalQA](https://github.com/haven-jeon/LegalQA)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    j = response.json()
    if j['data'][0]['matches'] is None:
        return {}
    total_result = []
    if j['data'][0]['tags']['ai_response'] == '':
        total_result.append('죄송합니다. 답변을 할 수 없습니다.\n다른 질문을 해주세요!')
    else:
        total_result.append(j['data'][0]['tags']['ai_response'])
    for m in j['data'][0]['matches']:
        m['chunks']= None
        total_result.append({'id': m['id'], 'scores': m['scores']['cosine']['value'], 'text': m['text'], 'tags': m['tags']})
    return total_result

def get_text():
    input_text = st.text_input("You: ","부의금은 누구에게 귀속되는지?", key="input")
    return input_text 


user_input = get_text()

if user_input:
    output = query({
                "data": [
                    {
                    "text": user_input
                    }
                ],
                "targetExecutor": "",
                "parameters": {"limit": 3}
                })

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i][0], key=str(i), avatar_style='adventurer-neutral')
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        st.json(st.session_state["generated"][i][1:], expanded=False)