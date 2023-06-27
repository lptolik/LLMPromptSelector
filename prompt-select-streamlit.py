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
    verbose=True
)

gpt4 = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.0,
    verbose=True
)

# Initialise session state variables:
st.session_state["vectors"] = []
st.session_state["prompt_chosen"] = False
st.session_state["chat_history"] = []
st.session_state["chat_interactions"] = []
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

# PROMPT EDITOR: edits the identified prompt to best use:
# @st.cache_resource
# def edit_prompt(chosen_prompt):
#     progress_bar = st.progress(0, "Initialising editing model...")
    
#     # Create a template for answering the user query using the selected prompt
#     prompt_editor_temp = """You are a prompt editing system. 
#     You must identify the part of the given prompt where the first user input/suggestion/request/code/object is specified and return all of the prompt except for the identified part and any text relating to the first user input. Notice that anything talking about "your" objective shouldnt be changed because its not user input. You must also
#     replace all curly brackets with some other character. Finally, you should make the text as readable for a human as possible so remove special characters like back slashes that dont add anything useful to the prompt. Leave the rest of the prompt exactly as it is. Return
#     only the edited prompt without any explanations. Do not answer any questions or add anything.
#     The prompt you must edit: 
#     {chosen_prompt}'."""

#     edit_prompt = PromptTemplate(
#         input_variables=["chosen_prompt"],
#         template = prompt_editor_temp
#     )

#     editing_chain = LLMChain(llm = gpt4,    # can be changed to 3.5 but Old Man gets through :/
#                     prompt = edit_prompt,
#                     verbose=True)
    
#     progress_bar.progress(25, "Editing in progress... Please wait...")
    
#     print(f"THIS IS CHOSEN PROMPT: {chosen_prompt}")
#     final_prompt = editing_chain.run(chosen_prompt)
#     progress_bar.progress(50, "Initializing settings model...")
    
#     print(f"This is the edited prompt: {final_prompt}")
    
#     settings_prompt_template = """
#     Act as a GPT expert that chooses the best settings for a large language model based on a provided prompt 
#     it will use. You must choose the best fitting temperature, top-p, presence penalty and frequency penalty 
#     settings based on how creative, factual and relevant the answer to the provided prompt should be. For each parameter you must choose a value between 0.0 and 1.0 A high 
#     temperature or top p value produces more unpredictable and interesting results, but also increases the 
#     likelihood of errors or nonsense text. A low temperature or top p value can produce more conservative and 
#     predictable results, but may also result in repetitive or uninteresting text. For text generation tasks, 
#     you may want to use a high temperature or top p value. However, for tasks where accuracy is important, 
#     such as translation tasks or question answering, a low temperature or top p value should be used to 
#     improve accuracy and factual correctness. The presence penalty and frequency penalty settings are useful 
#     if you want to get rid of repetition in your outputs. Do not be afraid of using the entire scale 
#     for each parameter. Try to stay away from values in the middle of the scale unless you need to. Provide a succint explanation on why and how you chose each parameter. Provide
#     the settings that you choose in a dictionary format.
#     Your output should be the explanation of your choices followed by the dictionary of settings.
    
#     The provided prompt: 
#     {final_prompt}
#     """
    
#     settings_prompt = PromptTemplate(
#         input_variables=["final_prompt"],
#         template=settings_prompt_template
#     )
    
#     settings_chain = LLMChain(
#         llm = gpt4,
#         prompt=settings_prompt,
#         verbose=True
#     )
    
#     progress_bar.progress(60, "Configuring settings...")
#     explanation, settings = extract_dictionary(settings_chain.run(final_prompt))
#     progress_bar.progress(100, "Done!")
#     print(f"Settings reasoning:\n{explanation}")
#     print(f"\n{settings}")
    
#     return final_prompt, explanation, settings

def initialise_editing_model():
    # Create a template for answering the user query using the selected prompt
    prompt_editor_temp = """You are a prompt editing system. 
    You must identify the part of the given prompt where the first user input/suggestion/request/code/object is specified and return all of the prompt except for the identified part and any text relating to the first user input. Notice that anything talking about "your" objective shouldnt be changed because its not user input. You must also
    replace all curly brackets with some other character. Finally, you should make the text as readable for a human as possible so remove special characters like back slashes that dont add anything useful to the prompt. Leave the rest of the prompt exactly as it is. Return
    only the edited prompt without any explanations. Do not answer any questions or add anything.
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
    settings based on how creative, factual and relevant the answer to the provided prompt should be. For each parameter you must choose a value between 0.0 and 1.0 A high 
    temperature or top p value produces more unpredictable and interesting results, but also increases the 
    likelihood of errors or nonsense text. A low temperature or top p value can produce more conservative and 
    predictable results, but may also result in repetitive or uninteresting text. For text generation tasks, 
    you may want to use a high temperature or top p value. However, for tasks where accuracy is important, 
    such as translation tasks or question answering, a low temperature or top p value should be used to 
    improve accuracy and factual correctness. The presence penalty and frequency penalty settings are useful 
    if you want to get rid of repetition in your outputs. Do not be afraid of using the entire scale 
    for each parameter. Try to stay away from values in the middle of the scale unless you need to. Provide a succint explanation on why and how you chose each parameter. Provide
    the settings that you choose in a dictionary format.
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
# Display the prompts table:
display_prompts()

st.sidebar.title("Sidebar")
st.session_state["model"] = st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo", "gpt-4"))
st.session_state["show_details"] = st.sidebar.checkbox("Show details!")

if "already_displayed" not in st.session_state:
    st.session_state["already_displayed"] = False

if not st.session_state["show_details"] and st.session_state["already_displayed"]:
    st.session_state["already_displayed"] = False

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
    settings_explanation, settings = extract_dictionary(settings_chain.run(final_prompt))
    progress_bar.progress(100, "Done!")
    
    #final_prompt, settings_explanation, settings = edit_prompt(chosen_prompt)
        
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

# Display form to select prompt:
with st.form("form"):
    prompt_df = retrieve_best_prompts(db, user_input)
    
    # Allow the user to select what prompt to use:
    chosen_act = st.selectbox("Prompt", (prompt_df["act"]))
    chosen_prompt = prompt_df.loc[prompt_df.act == chosen_act].prompt.values[0]

    # Submit user selection:
    submit = st.form_submit_button("Submit choice")
    if submit:    
        # Display chosen prompt og and edited version to user:
        st.session_state["chosen prompt"] = chosen_prompt
        st.session_state["prompt_chosen"] = True
        
        final_prompt, settings, settings_explanation = edit_and_configure_prompt(chosen_prompt)
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