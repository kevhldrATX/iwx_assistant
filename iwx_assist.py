import streamlit as st
import yaml
from gradio_client import Client
import openai

mode = "combo"

st.set_page_config(page_title="Infoworks AI", page_icon="🔥", layout="wide")

# Initialize openAI
openai.api_key = st.secrets["openai_API_KEY"]


if mode != "general":
    client = Client(st.secrets["gradio_link"])


def complete(prompt, option):
    if option != 'General' and mode != 'general':
        if option == 'API Help':
            action = 'API HELP'

        if option == 'Product Help':
            action = 'Docs'

        if option == 'Code':
            action = 'Code'

        if option == 'Command':
            action = 'API'

        try:
            res = client.predict(
                        action,	# str (Option from: ['API', 'Docs', 'Code']) in 'Choose question Type' Dropdown component
                        prompt,	# str in 'User Question' Textbox component
                        api_name="/predict"
                )
        except:
            res = "I do not have an answer to that request."
    else:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ])['choices'][0]['message']['content']
        except:
            res = "I do not have an answer to that request."
        

    return res

st.image('https://www.infoworks.io/wp-content/uploads/2022/09/logo-orig.svg')

option = st.radio('', ('Code', 'API Help', 'Product Help', 'General'), horizontal=True)

st.write("#### Infoworks Assistant")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

message1 = "Hello, how can I help you?"

with st.chat_message("assistant"):
    st.write(message1)

for chat in st.session_state['chat_history']:
    with st.chat_message(chat[0]):
        if chat[2] == "code":
            st.code(chat[1])
        else:
            st.write(chat[1])

chat_container = st.container()

search_criteria = chat_container.chat_input()

# if search_criteria := st.chat_input():

if search_criteria:
    
    st.chat_message("user").write(search_criteria)

    st.session_state.chat_history.append(("user", search_criteria, "text"))

    with st.spinner():
        answer = complete(search_criteria, option)

        with st.chat_message("assistant"):
            
            if answer.find('```') != -1:
                st.write(answer)
            else:
                st.code(answer, language='python')

            st.session_state.chat_history.append(("assistant", answer, "code"))