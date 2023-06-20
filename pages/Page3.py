import streamlit as st
import pandas as pd
import os
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

OPENAI_API_KEY = "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"
WEAVIATE_URL = "https://first-test-cluster-dw7v1rzb.weaviate.network"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = OpenAI(temperature=0, verbose=True)

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
    template = final_prompt + list_to_string() + default_temp

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
else:
    # The user will interact with the selected prompt:
    print("PROMPT IS GIVEN:")
    if "selected" not in st.session_state:  # Making sure the 'default' value is not updated after first load
        st.session_state.selected_prompt_val = f'The prompt I will use is: \n{st.session_state["final_prompt"]}'
        
        llm_chain = update_interactive_llm(st.session_state["final_prompt"])
        value = llm_chain.predict(user_input=st.session_state["user input"])
        
        st.session_state["selected"] = value
        st.session_state["default"] = st.session_state["selected"]
        
        if st.session_state["chat_history"] == []:  # This might be meaningless.... TODO
            st.session_state["chat_history"].append(f"\n\nUser: {st.session_state['user input']} ")
            st.session_state["chat_history"].append(f" \nSystem: {value} ")

# Form for interacting with the llm:
with st.form("llm-interaction"):
    st.text_area("Selected prompt:", key="selected-prompt", height=200, value=st.session_state.selected_prompt_val)
    st.text_area("LLM output:", key="output", height=200, value=st.session_state["default"])
    
    user_response=st.text_input("User response", key="user-response")
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        llm_chain = update_interactive_llm(st.session_state["final_prompt"])
        system_response = llm_chain.predict(user_input=user_response)
        
        st.session_state["chat_history"].append(f"\n\nUser: {user_response}")
        st.session_state["chat_history"].append(f"\nSystem: {system_response}")
        
        st.text_area("LLM output:", value=system_response)
        
        print(st.session_state["chat_history"])