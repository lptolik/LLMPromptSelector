import streamlit as st
import pandas as pd
import os
import re
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
import pinecone

OPENAI_API_KEY = "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"
WEAVIATE_URL = "https://first-test-cluster-dw7v1rzb.weaviate.network"

PINECONE_API_KEY = "cf10ee0c-8558-45e2-82b2-864bea80d179"
PINECONE_ENV = "us-west4-gcp-free"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embeddings = OpenAIEmbeddings()
#llm = OpenAI(temperature=0, verbose=True)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0,
    top_p = 0,
    verbose=True
)

gpt4 = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.0,
    top_p = 0.0,
    verbose=True
)

# Initialise session state variables:
st.session_state["vectors"] = []
st.session_state["prompt_chosen"] = False
st.session_state["chat_history"] = []   # Keeps track of all user and system messages as a list of strings. Used to provide chat history to models. Probably should eventually be removed and replaced with "interactions"
st.session_state["chat_interactions"] = []  # Keeps track of User/System messages as a list of dictionaries [{role: "", content: ""}]. Used to display messages in chat style.
if "chat_archive" not in st.session_state:
    st.session_state["chat_archive"] = []   # Keeps track of past chat conversations that are saved
st.session_state["model"] = "gpt-3.5-turbo"
st.session_state["final_prompt"] = ""
st.session_state["settings"] = {"temperature": 0.0, "top_p": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
st.session_state["user input"] = ""
st.session_state.selected_prompt_val = f"I will use my default settings to answer your questions."
if 'sys_msgs' not in st.session_state:
    st.session_state['sys_msgs'] = []
if 'usr_msgs' not in st.session_state:
    st.session_state['usr_msgs'] = []
if "default" in st.session_state:
    del st.session_state["default"]
if "selected" in st.session_state:
    del st.session_state["selected"]
if "llm_chain" in st.session_state:
    del st.session_state["llm_chain"]
if "agent_enabled" in st.session_state:
    del st.session_state["agent_enabled"]
if "saved_chat" not in st.session_state:
    st.session_state["saved_chat"] = ""
    

# Assumes the first use of curly brackets is for the dictionary:
def extract_dictionary(text):
    stack = []

    for m in re.finditer(r'[{}]', text):
        pos = m.start()

        if m.group() == '{':
            stack.append(pos)
        elif stack:
            start_pos = stack.pop()
            if not stack:
                return (text[:start_pos], text[start_pos:pos + 1])
    return ""
    
# display the prompts in a table:
@st.cache_data
def display_prompts():
    prompt_df = pd.read_csv('prompts_data_updated.csv')
    st.write(prompt_df)
    
# Load the .csv database of prompts:
@st.cache_resource
def load_prompts():
    loader = CSVLoader("prompts_data_updated.csv", csv_args={
        'fieldnames': ['Act', 'Prompt']
    })
    documents = loader.load()

    # Create the vectorstore using Weaviate and documents
    st.session_state["vectors"] = Weaviate.from_documents(documents, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)
    return st.session_state["vectors"]

@st.cache_resource
def load_pinecone_prompts():
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )

    index_name="promptsvectors"
    
    st.session_state["vectors"] = Pinecone.from_existing_index(index_name, embeddings)
    return st.session_state["vectors"]

# RETRIEVER, finds the most relevant prompt and returns it:
def retrieve_best_prompts(db, user_input):
    potential_prompts = db.similarity_search(user_input, k=4)
    potential_prompts_dict = []
    # Display the potential prompts to the user:
    for document in potential_prompts:
        # Split the string into two parts using the "prompt: " as the separator
        act, prompt = document.page_content.split("\nprompt: ")
        # Strip the "act: " and "prompt: " from the respective parts
        act = act.replace("act: ", "").strip()
        prompt = prompt.strip()
        potential_prompts_dict.append({"act": act, "prompt": prompt})
    prompt_df = pd.DataFrame(potential_prompts_dict)
    st.write(prompt_df)
    
    return prompt_df

def initialise_linking_model():
    linker_temp = """
    You are an expert linking system, acting as the link between two large language models. You are given the chat history of a user with the first llm and the prompt the second llm will be using for interaction with the same user. Your goal is to alter the provided chat history so it can be provided to the second llm to use as additional information. This alteration could involve summarizing the chat, removing every part of the chat other than the first/last sections, or even extracting specific information from the chat. 
    For example: suppose you are given the chat history of a user planning a trip to London, the chat history has mention of different locations, dates and other relevant information to planning a trip. You are then given the prompt the second llm will use and it is a python developer prompt. You should extract all the important trip information and present it in a clear format so the Python llm can use it to perform its actions such as visualising the locations/routes on a map.
    Another example: suppose you are given the chat history of a user that is asking for an overview of a topic or a summarization of some text, you are then given the prompt the second llm will use which is a "Translator" prompt. You would want to only keep the system responses that are the answers to the users questions. Most likely only the final system response containing the final answer should be kept.
    Dont summarise things like essays or reports.
    Your output is only the altered chat history.

    Chat history: 
    {chat_history}
    
    Second llm prompt:
    {prompt}
    """
    
    linker_prompt = PromptTemplate(
        input_variables=["chat_history", "prompt"],
        template=linker_temp
    )
    
    linker_chain = LLMChain(
        llm=llm,
        prompt=linker_prompt,
        verbose=True
    )
    
    return linker_chain

def initialise_editing_model():
    # Create a template for answering the user query using the selected prompt
    prompt_editor_temp = """
    You are a prompt editing system. 
    You must identify the part of the given prompt where the first user input/suggestion/request/code/object is specified and return all of the prompt except for the identified part and any text relating to the first user input. The first user input is usually in quotes. Notice that anything talking about "your" objective shouldnt be changed because its not user input. 
    If the prompt specifies 'how' you should respond or what you should provide you must leave that intact. There is a difference between a statement saying 'how' you should respond and an actual user input. It might be the case that you dont need to remove anything. Finally, you should make the text as readable for a human as
    possible so remove special characters like back slashes that dont add anything useful to the prompt. Leave the rest of the prompt exactly as it is. Return only the edited prompt without any explanations. Do not answer any questions or add anything.
    The prompt you must edit: 
    {chosen_prompt}'."""

    edit_prompt = PromptTemplate(
        input_variables=["chosen_prompt"],
        template = prompt_editor_temp
    )

    editing_chain = LLMChain(llm = gpt4,    # can be changed to 3.5 but Old Man gets through :/
                    prompt = edit_prompt,
                    verbose=True)
    
    return editing_chain

def initialise_settings_model():
    settings_prompt_template = """
    Act as a GPT expert that chooses the best settings for a large language model based on a provided prompt 
    it will use. You must choose the best fitting temperature, top-p, presence penalty and frequency penalty 
    settings based on how creative, factual and relevant the answer to the provided prompt should be. For each parameter you must choose a value between 0.0 and 1.0. A high 
    temperature or top p value produces more unpredictable and interesting results, but also increases the 
    likelihood of errors or nonsense text. A low temperature or top p value can produce more conservative and 
    predictable results, but may also result in repetitive or uninteresting text. For text generation tasks, 
    you may want to use a high temperature or top p value. However, for tasks where accuracy is important, 
    such as translation tasks or question answering, a low temperature or top p value should be used to 
    improve accuracy and factual correctness. The presence penalty and frequency penalty settings are useful 
    if you want to get rid of repetition in your outputs. You can think of Frequency Penalty as a way to prevent word repetitions, 
    and Presence Penalty as a way to prevent topic repetitions. You can use the entire scale 
    for each parameter. Consider your choices carefully and perform multiple iterations before deciding on a final answer. 
    The settings you choose should be the most likely settings so that if run again you would select the same exact settings again. 
    Provide a succint explanation on why and how you chose each parameter. Provide the settings that you choose in a dictionary format.
    Your output should be the explanation of your choices followed by the dictionary of settings.
    
    The provided prompt: 
    {final_prompt}
    """
    
    settings_prompt = PromptTemplate(
        input_variables=["final_prompt"],
        template=settings_prompt_template
    )
    
    settings_chain = LLMChain(
        llm = gpt4,
        prompt=settings_prompt,
        verbose=True
    )
    return settings_chain
    
#-------------------------------------------------------------------------------------
#---------------------DISPLAY STREAMLIT OBJECTS:--------------------------------------
#-------------------------------------------------------------------------------------
st.title("Prompt Selector")

# Display the prompts table:
display_prompts()

st.sidebar.title("Options:")
st.session_state["model"] = st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo", "gpt-4"))
st.session_state["show_details"] = st.sidebar.checkbox("Show details!")

# Display saved chats:
if len(st.session_state["chat_archive"]) > 0:
    # Truncate to most recent 3 chats:
    if len(st.session_state["chat_archive"]) > 3:
        st.session_state["chat_archive"] = st.session_state["chat_archive"][-3:]
    st.sidebar.title("Previous chats:")
    archive_df = pd.DataFrame.from_dict(st.session_state["chat_archive"])
    st.sidebar.write(archive_df)
    new_row = {"Chat name": "None", "Conversation": ""}
    new_archive_df = archive_df.append(pd.DataFrame([new_row], index=['0'], columns=archive_df.columns))
    saved_chat_name = st.sidebar.selectbox("Use chat", options=(new_archive_df["Chat name"][::-1]))
    if saved_chat_name != "None":
        st.session_state["saved_chat"] = new_archive_df.loc[new_archive_df["Chat name"] == saved_chat_name].conversation.values[0]
    else:
        st.session_state["saved_chat"] = "" 

async def async_generate(chain, chosen_prompt):
    resp = await chain.arun(chosen_prompt)
    print(resp)
    return resp

async def run_concurrent(chosen_prompt, settings_chain, editing_chain):
    tasks = [async_generate(editing_chain, chosen_prompt), async_generate(settings_chain, chosen_prompt)]
    return await asyncio.gather(*tasks)

#@st.cache_resource
def link_chat(final_prompt):
    linker_chain = initialise_linking_model()
    altered_chat = linker_chain.run(chat_history=st.session_state["saved_chat"], prompt=final_prompt)
    final_prompt = f"{final_prompt} \n\nYou also have the following information from a previous interaction with the user. You may use this when responding: '{altered_chat}'"
    return final_prompt

@st.cache_resource
def edit_and_configure_prompt(chosen_prompt):
    progress_bar = st.progress(0, "Initialising editing model...")
        
    # EDIT THE PROMPT:
    editing_chain = initialise_editing_model()
    progress_bar.progress(25, "Editing in progress... Please wait...")
    final_prompt = editing_chain.run(chosen_prompt)
    
    # CONFIGURE SETTINGS:
    progress_bar.progress(50, "Initializing settings model...")
    settings_chain = initialise_settings_model()
    progress_bar.progress(60, "Configuring settings...")
    settings_explanation, settings = extract_dictionary(settings_chain.run(chosen_prompt))
    progress_bar.progress(100, "Done!")
    
    # progress_bar = st.progress(0, "Initialising editing model...")
    # editing_chain = initialise_editing_model()
    # progress_bar.progress(15, "Initialising settings model...")
    # settings_chain = initialise_settings_model()
    # progress_bar.progress(40, "Running editing and settings model...")
    # final_prompt, settings_chain_res = asyncio.run(run_concurrent(chosen_prompt, settings_chain, editing_chain))
    # progress_bar.progress(75, "Extracting dictionary...")
    # settings_explanation, settings = extract_dictionary(settings_chain_res)
    # progress_bar.progress(100, "Done!")
        
    return final_prompt, settings, settings_explanation

# Display the container to enter prompt query:
with st.container():
    db = load_pinecone_prompts()
    #db = load_prompts()
    user_input = st.text_input("Enter query", key="query")
    if user_input == "":    # Upon first load stop here
        st.stop()
        
    st.session_state["user input"] = user_input
    
    query_submit = st.button("Submit Query")
    if query_submit:
        st.write("Now head on over to the 'Interact-with-the-llm' page to interact with the LLM using the selected prompt.")

import asyncio

# Display form to select prompt:
with st.form("form"):
    prompt_df = retrieve_best_prompts(db, user_input)
    
    # Allow the user to select what prompt to use:
    chosen_act = st.selectbox("Prompt", (prompt_df["act"]))
    chosen_prompt = prompt_df.loc[prompt_df.act == chosen_act].prompt.values[0]
    
    # final_prompt, settings, settings_explanation = "", "", ""
    # async def retrieve_prompt_and_settings():
    #     final_prompt, settings, settings_explanation = await edit_and_configure_prompt(chosen_prompt)

    # Submit user selection:
    submit = st.form_submit_button("Submit choice")
    if submit:    
        # Display chosen prompt og and edited version to user:
        st.session_state["prompt_chosen"] = True
        final_prompt, settings, settings_explanation = edit_and_configure_prompt(chosen_prompt)
        
        if st.session_state["saved_chat"] != "":
            final_prompt = link_chat(final_prompt)
        
        st.session_state["final_prompt"] = final_prompt
        st.session_state["settings"] = json.loads(settings)
        
        if st.session_state["show_details"]:
            st.subheader(f"Best fitting prompt found:")
            st.write(f"\n\n{chosen_prompt}")
            
        st.subheader(f"The prompt after editing:")
        st.write(f"\n\n{final_prompt}")
        
        if st.session_state["show_details"]:
            st.subheader(f"Settings reasoning:")
            st.write(f"\n{settings_explanation}")
            st.write(f"{json.loads(settings)}")