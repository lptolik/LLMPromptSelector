import streamlit as st
import pandas as pd
import os
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message

OPENAI_API_KEY = "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"
WEAVIATE_URL = "https://first-test-cluster-dw7v1rzb.weaviate.network"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#llm = OpenAI(temperature=0, verbose=True)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name=st.session_state["model"],
    temperature=st.session_state["settings"]["temperature"],
    frequency_penalty=st.session_state["settings"]["frequency_penalty"],
    presence_penalty=st.session_state["settings"]["presence_penalty"],
    top_p=st.session_state["settings"]["top_p"],
    verbose=True
)

default_temp = """

User: {user_input}
System:"""

# Helper function to convert list of strings that serve as the chat history into a large string:
def list_to_string():
    output = ""
    for i in st.session_state["chat_history"]:
        output += i
    return output

# Initialises/updates the llm and the prompt it uses:
def update_interactive_llm(final_prompt):
    template = "You must NEVER generate a 'user' response yourself" + final_prompt + list_to_string() + default_temp

    prompt = PromptTemplate(
        #input_variables=["chat_history", "user_input"],
        input_variables=["user_input"],
        template=template
    )
    
    print(f"This is the prompt: {template}")

    llm_chain = LLMChain(
        llm=llm,
        prompt = prompt,
        verbose=True,
    )
    return llm_chain

# Setting the settings of the llm, depending on whether a prompt has been selected or not
if not st.session_state["prompt_chosen"]:
    # If a prompt has not been selected then the user will interact with default llm:
    print("PROMPT NOT CHOSEN:")
    if "default" not in st.session_state:   # Making sure the 'default' value is not updated after first load
        llm_chain = update_interactive_llm("")
        value = llm_chain.predict(user_input="")
        st.session_state["default"] = value
        st.session_state["chat_history"].append(f" \nSystem: {value} ")
        st.session_state["chat_interactions"].append({"role": "system", "content": value})
else:
    # The user will interact with the selected prompt:
    print("PROMPT IS GIVEN:")
    if "selected" not in st.session_state:  # Making sure the 'default' value is not updated after first load
        st.session_state.selected_prompt_val = f'The prompt I will use is: \n{st.session_state["final_prompt"]}\n The settings are: {st.session_state["settings"]}\n The model being used is: {st.session_state["model"]}'
        
        llm_chain = update_interactive_llm(st.session_state["final_prompt"])
        value = llm_chain.predict(user_input=st.session_state["user input"])
        
        st.session_state["selected"] = value
        st.session_state["default"] = st.session_state["selected"]
        
        if st.session_state["chat_history"] == []:  # This might be meaningless.... TODO
            st.session_state["chat_history"].append(f"\n\nUser: {st.session_state['user input']} ")
            st.session_state["chat_interactions"].append({"role": "user", "content": st.session_state['user input']})
            st.session_state["chat_history"].append(f" \nSystem: {value} ")
            st.session_state["chat_interactions"].append({"role": "system", "content": value})

# Form for interacting with the llm:
# with st.form("llm-interaction"):
#     st.text_area("Selected prompt:", key="selected-prompt", height=200, value=st.session_state.selected_prompt_val)
#     st.text_area("LLM output:", key="output", height=200, value=st.session_state["default"])
    
#     user_response=st.text_input("User response", key="user-response")
    
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         llm_chain = update_interactive_llm(st.session_state["final_prompt"])
#         system_response = llm_chain.predict(user_input=user_response)
        
#         st.session_state["chat_history"].append(f"\n\nUser: {user_response}")
#         st.session_state["chat_interactions"].append({"role": "user", "content": user_response})
#         st.session_state["chat_history"].append(f"\nSystem: {system_response}")
#         st.session_state["chat_interactions"].append({"role": "system", "content": system_response})
        
#         st.text_area("LLM output:", value=system_response)
#         display_chat_interaction()
        
#         print(st.session_state["chat_interactions"])

st.text_area("Selected prompt:", key="selected-prompt", height=200, value=st.session_state.selected_prompt_val)
   
response_container = st.container()

container = st.container()   

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_response = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_response:
        llm_chain = update_interactive_llm(st.session_state["final_prompt"])
        system_response = llm_chain.predict(user_input=user_response)
        
        st.session_state["chat_history"].append(f"\n\nUser: {user_response}")
        st.session_state["chat_interactions"].append({"role": "user", "content": user_response})
        st.session_state["chat_history"].append(f"\nSystem: {system_response}")
        st.session_state["chat_interactions"].append({"role": "system", "content": system_response})
        
        print(st.session_state["chat_interactions"])

if st.session_state['chat_interactions']:
    with response_container:
        for i in range(len(st.session_state['chat_interactions'])):
            if st.session_state['chat_interactions'][i]["role"] == "user":
                message(st.session_state["chat_interactions"][i]["content"], is_user=True, key=str(i) + '_user')
            else:
                message(st.session_state["chat_interactions"][i]["content"], key=str(i))