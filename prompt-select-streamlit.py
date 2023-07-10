import streamlit as st
import pandas as pd
import os
import re
import json
import asyncio
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

#-------------------------------------------------------------------------------------
#---------------------DEFINE STREAMLIT SESSION VARIABLES:-----------------------------
#-------------------------------------------------------------------------------------

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
    
#-------------------------------------------------------------------------------------
#----------------------STORED PROMPT PROCESSING:--------------------------------------
#-------------------------------------------------------------------------------------

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

# As above function, but using Pinecone DB
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

#-------------------------------------------------------------------------------------
#-----------------------------DEFINE LLM CHAINS:--------------------------------------
#-------------------------------------------------------------------------------------

# Defines the chain that judges the relevance of a chat_history message to a new conversation
def initialise_relevance_rating_model():
    rating_temp = """
    You are an AI system specializing in relevance analysis. Your task is to assess the relevance of a past message exchange between a User and a System in relation to a new conversation. For instance, if the past conversation was about writing an essay and the new conversation is about translation, the message exchange containing the essay would likely be most relevant as that is probably what the user wants to translate. Similarly, if a previous conversation was about creating and refining an itinerary and a new conversation is about visualising a route on a map using some tool, then the final itinerary created in the past conversation is likely the most relevant to the new conversation.
    For the new conversation, you will be given the initial User query and the System's response prompt. Based on these inputs, you are required to rate the relevance of the past message exchange on a scale of 1 to 10. A rating of 1 signifies that the past exchange is completely irrelevant or not at all useful to the new conversation, while a rating of 10 indicates that the past exchange is highly relevant or extremely useful to the new conversation.
    Notice, messages dont need to be related to the new conversation topic or query to be relevant because past messages were created without any knowledge of the new prompt or new query. You must instead consider if the content of the past message exchange contains information that could be useful to the new conversation.
    Here are the details you need to consider:

        Past Message Exchange:
        {message}
        
        New Conversation Query:
        {query}
        
        System's Response Prompt for the New Conversation:
        {prompt}
        
    Please provide your relevance rating in square brackets and include a brief explanation for your rating.
    Your output format must be: 
    "[rating]
    
    Explanation..."
    """
    
    rating_prompt = PromptTemplate(
        input_variables=["message", "query", "prompt"],
        template=rating_temp
    )
    
    rating_chain = LLMChain(
        llm=llm,
        prompt=rating_prompt,
        verbose=True
    )
    
    return rating_chain

# Defines the LLM chain that alters relevant messages to best fit a new prompt
def initialise_linking_model():
    linker_temp = """
    You are an expert linking system, acting as the link between two large language models. You are given the most relevant messages from a past conversation of a User with the first llm, a new User query and the prompt the second llm will be using for interaction with the same User. Your goal is to edit the provided chat history so it can be provided to the second llm to use as additional information when answering the new User query. This editing could involve removing parts of the chat or extracting specific information from the chat. 
    For example: suppose you are given the chat history of a user planning a trip to London, the chat history has mention of different locations, dates and other relevant information to planning a trip. You are then given the new User query "I want to visualise my route using python" and the prompt the second llm will use is a python developer prompt. You should extract all the important trip information and present it in a clear format so the second llm can use it to perform its actions such as visualising the locations/routes on a map.
    Another example: suppose you are given the chat history of the User that is asking for an overview of a topic, the User then enters a query "I want to translate my overview", you are then given the prompt the second llm will use which is a "Translator" prompt. You would want to only keep the the topic overview from the chat so that it can then be immediately translated.
    Another example: suppose you are given the chat history of the User that is writing an essay, the new User query is "I want to summarize my essay", and the prompt for the second llm is a "Summarizer" prompt. You MUST NOT summarise yourself. You must only take the essay itself from the chat history (while dropping User messages) so that the second llm can immediately summarise it.
    Some prompts will require you to extract information/summarise information from the chat history to provide as context, while others, like an essay writer prompt or a translator prompt, will require you to keep the System messages answering User requests intact and unaltered. It is up to you to decide what you do to the chat history, but carefully consider all the angles of how the chat history could be useful for the next prompt before changing any information.
    
    Relevant messages: 
    {chat_history}
    
    User query:
    {query}
    
    Second llm prompt:
    {prompt}
    
    Consider first what the User is trying to achieve based on their query.
    Next, consider what the aim of the provided prompt is.
    You must then consider what information from the relevant messages can help the User achieve their goal, using the provided prompt.
    
    NEVER EVER DO THE JOB OF THE SECOND LLM PROMPT YOURSELF!!!
    
    Your only output is the information you extract from the relevant messages, without the original messages themselves.
    Place the information that should be provided to the second llm into square brackets.
    """
    
    linker_prompt = PromptTemplate(
        input_variables=["chat_history", "query", "prompt"],
        template=linker_temp
    )
    
    
    linker_chain = LLMChain(
        llm=gpt4,
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
    if len(st.session_state["chat_archive"]) > 6:
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

async def async_generate(chain, message, final_prompt):
    resp = await chain.arun(message=message, query=st.session_state["user input"], prompt=final_prompt)
    print(resp)
    return (message, resp)

async def run_concurrent_rating(rating_chain, messages, final_prompt):
    tasks = [async_generate(rating_chain, message, final_prompt) for message in messages]
    return await asyncio.gather(*tasks)

# Processes the chat history to provide context to new conversation:
@st.cache_resource
def link_chat(final_prompt, context_strictness):
    progress_bar = st.progress(0, "Initialising relevance detection model...")
    rating_chain = initialise_relevance_rating_model()
    messages = st.session_state["saved_chat"].split("User:")[1:]
    
    progress_bar.progress(25, "Finding most relevant messages...")
    ratings = asyncio.run(run_concurrent_rating(rating_chain, messages, final_prompt))
    
    # Display message relevance:
    with st.container():
        st.write("Most relevant messages identified:")
        st.write(pd.DataFrame.from_records(ratings))
    
    # Helper function
    def extract_number(text):
        number = text.partition("[")[2].partition("]")[0]
        if number == "":
            number = "0"
        return int(number)
    
    # If relevance cut-off too high - pick the maximum relevance messages:
    max_rating = max(extract_number(rating) for (message, rating) in ratings)
    if context_strictness > max_rating:
        context_strictness = max_rating
    
    messages_to_keep = [message for (message, rating) in ratings if extract_number(rating) >= context_strictness]
    
    progress_bar.progress(50, "Initialising linking model...")
    linker_chain = initialise_linking_model()
    
    progress_bar.progress(60, "Running linking model...")
    final_context = linker_chain.run(chat_history=messages_to_keep, prompt=final_prompt, query=st.session_state["user input"])
    final_prompt = f"{final_prompt} \n\n#CONTEXT: '{final_context}' Use this if the user refers to #CONTEXT"

    progress_bar.progress(100, "Finished linking chat!")
    return final_prompt, ratings

@st.cache_resource
def edit_and_configure_prompt(chosen_prompt):
    progress_bar = st.progress(0, "Initialising editing model...")
    editing_chain = initialise_editing_model()
    
    # EDIT THE PROMPT:
    progress_bar.progress(25, "Editing in progress... Please wait...")
    final_prompt = editing_chain.run(chosen_prompt)
    
    progress_bar.progress(50, "Initializing settings model...")
    settings_chain = initialise_settings_model()
    
    # CONFIGURE SETTINGS:
    progress_bar.progress(60, "Configuring settings...")
    settings_explanation, settings = extract_dictionary(settings_chain.run(chosen_prompt))
    
    progress_bar.progress(100, "Finished loading prompt!")
    return final_prompt, settings, settings_explanation

# Display the container to enter prompt query:
with st.container():
    db = load_pinecone_prompts()
    #db = load_prompts()    # alternative option using weaviate
    
    user_input = st.text_input("Enter query", key="query")
    if user_input == "":    # Upon first load stop here
        st.stop()   
    st.session_state["user input"] = user_input

# Display form to select prompt:
with st.form("form"):
    prompt_df = retrieve_best_prompts(db, user_input)
    
    # Allow the user to select what prompt to use:
    chosen_act = st.selectbox("Prompt", (prompt_df["act"]))
    chosen_prompt = prompt_df.loc[prompt_df.act == chosen_act].prompt.values[0]

    if st.session_state["saved_chat"] != "":
        strictness = st.slider("Relevance cut-off:", 1, 10, 6)
    
    # Submit user selection:
    submit = st.form_submit_button("Submit choice")
    if submit:    
        # Display chosen prompt og and edited version to user:
        st.session_state["prompt_chosen"] = True
        final_prompt, settings, settings_explanation = edit_and_configure_prompt(chosen_prompt)
        
        if st.session_state["saved_chat"] != "":
            final_prompt, ratings = link_chat(final_prompt, strictness)
        
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
            st.subheader(f"Messages ratings:")
            st.write(pd.DataFrame.from_records(ratings))