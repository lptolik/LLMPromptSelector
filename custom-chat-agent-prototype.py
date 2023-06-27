import os
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFium2Loader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
WEAVIATE_URL = "https://first-test-cluster-dw7v1rzb.weaviate.network"
embeddings = OpenAIEmbeddings()

from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Global model used for both the agent and the prompt:
llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0,
        verbose=True
    )

# Defining the actual answering system:
prompt =  PromptTemplate(
            input_variables=["user_input", "context"],
            template="""Act as a friendly wise old man, giving advice learned from a long life. Use a friendly and familiar, yet authoritative and expert tone and provide anecdotes or metaphors with any advice you give. You like frequently making philosophical metaphors. You are quite self-reflective and somewhat of a philosopher. You have made mistakes in your past and so you have many regrets, you want to help others avoid doing the same. You seek to pass down everything youve learned to others so your responses often refer to your own experiences. Your objective is to give advice and guidance based on a user input. Structure your response in paragraphs.
            Use the following information to answer the question: "{context}"
            
            User: {user_input}
            System:"""
        )

selected_prompt_chain = LLMChain(llm=llm, prompt=prompt)

# main_tool = Tool(
#     name="Wise Old Man",
#     func=selected_prompt_chain.run,
#     description=(
#         'Useful for constructing the Final Answer. The action input is the exact user query and you can provide context. Always use this tool last.'
#     )
# )

tools_list = []
chat_history = []

def list_to_string():
    output = ""
    for i in chat_history:
        output += i
    return output

def split_data(file_name):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 20,
            length_function = len,
            add_start_index = True
        )
    
    if file_name[-4:] == ".csv" or file_name[-4:] == ".txt":
        with open(file_name) as f:
            csv = f.read()

        documents = text_splitter.create_documents([csv])
        
    if file_name[-4:] == ".pdf":
        loader = UnstructuredPDFLoader(file_name)
        documents = loader.load()   # not sure if I should split this for larger pdfs
        
    return documents

def init_qa(db):

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    
    return qa

def create_new_tool(db_name, qa, description):
    tools_list.append(Tool(
        name=db_name,
        func=qa.run,
        description=(
            description
        ),
        return_direct=True
    ))

# First, load the data:
def load_new_data(file_name, db_name, description):
    split = split_data(file_name)

    # create the vectorstore:
    db = Weaviate.from_documents(split, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)
    
    qa = init_qa(db)
    
    create_new_tool(db_name, qa, description)
    
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # # Format chat historyL
        # kwargs["chat_history"] = list_to_string()
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
import re
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
def update_template():
    
    template = """You must respond to the input to the best of your ability.
    You have access to the following tools to get relevant information/context to the question: 
    
    {tools}
    
    If you use one of [{tool_names}], you must use the following format:
    
    Query: the input you must respond to
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action, should be as close to the original user query as possible
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    IF YOU DONT USE ANY TOOL YOUR OUTPUT SHOULD BE EMPTY!! YOUR ONLY OUTPUT SHOULD BE: "N/A"

    Chat history:
    {chat_history}

    Begin!
    
    Query: {input}
    {agent_scratchpad}
    """
    
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools_list,
        input_variables=["input", "intermediate_steps", "chat_history"]
    )
    
    return prompt

def set_up():
    load_new_data("UniFAQdata.csv", "University FAQ Knowledge Base", 'useful for answering frequently asked questions about University.')
    #load_new_data("sample.csv", "Companies Knowledge Base", 'useful for answering questions regarding companies information')
    load_new_data("2023 - Student Induction.pdf", "Student Induction Knowledge Base", "useful for answering questions regarding new student intern questions")
    
    prompt = update_template()
    
    output_parser = CustomOutputParser()
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    tool_names = [tool.name for tool in tools_list]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools_list, verbose=True)
    return agent_executor

# Main:
agent_exec = set_up()

user_input = input()

while user_input != "":
    response = agent_exec.run({"input" : user_input, "chat_history": list_to_string()})#user_input, chat_history=list_to_string())
    chat_history.append(f"User query: {user_input}\n")
    chat_history.append(f"Agent response: {response}\n")
    print(f"This is the agent response: {response}")
    if response == "N/A":
        response=""
    print(selected_prompt_chain.predict(user_input=user_input, context=response))
    user_input = input()
    
    
