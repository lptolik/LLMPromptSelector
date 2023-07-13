# Prompt Selector Bot

## Name
Prompt selector.

## Introduction
Prompt engineering is an extremely powerful tool which can significantly effect the quality of a llm response. To see this consider jailbreaking - writing prompts which circumvent restrictions placed by OpenAI entirely. Additionally to writing style, structure and tone settings that can be defined by prompts, OpenAI also provides access to 4 settings that can alter the LLM response even further: temperature, top-p, presence penalty and frequency penalty. While some people might be familiar with temperature and top-p, very few have heard of the latter two. Good prompt writing and configuring of model settings is what can set a professional prompt engineer apart from the average chatGPT user, but prompt engineering is a skill and it takes time and a lot of trial and error to learn it.

The Prompt Selector app seeks to offer the average user the ability to make the most out of prompt engineering while not possessing the skill. This is done by the app selecting prompts and settings for you. The prompts are selected from a large database of pre-written prompts based on your initial query and settings are chosen on the fly based on the selected prompt. 

## Description
Main functionality: selects a "professional" prompt from a list of pre-defined prompts for the user to use based on their input. Additionally selects the best suited parameters for the chosen prompt to get the best results. Saves the average user time writing a prompt from scratch/finding a prompt from a large list of existing ones as well as selecting the most appropriate model parameters. Under the assumption that the average user is not familiar with "Prompt engineering techniques" or what makes a good prompt, this program automates prompt selection and configuration.

Since the prompt is selected from existing written ones stored in "prompts_data_updated.csv", potentially nonsensical llm-generated prompts are avoided.

Additional functionality: Context upload.
Context, in the form a .pdf, .html, .csv, .txt file can be uploaded, which will then be used as a knowledge base to respond to user queries. This feature combats hallucinations by allowing the user to provide their own trusted information.

Additional functionality: Web search
Access to the web can be enabled to allow real-time up to date context gathering.

Additional functionality: Linking chats
If the user chooses to save their chat, they then have the option of linking the saved chat to a new conversation. The most relevant messages from their saved conversation will be extracted and edited to fit a new conversation.

## Visuals
Front page:
![Front page, prompt selection:](/screenshots/Page1-upon-load.png?raw=true "Front page")

Prompt selection:
![Front page, prompt selection:](/screenshots/Page1-upon-prompt-selection.png?raw=true "Front page")

Prompt configuration, no details:
![Front page, prompt configuring:](/screenshots/Page1-upon-prompt-configuring-no-details.png?raw=true "Cofiguring prompt")

Prompt configuration, show details option:
![Front page, prompt configuration with details:](/screenshots/Page1-upon-prompt-configuring-yes-details.png?raw=true "Congifuring prompt and showing reasoning")

Selecting saved chat:
![Front page, Selecting saved chat:](/screenshots/Page1-choose-saved-chat.png?raw=true "Selecting saved chat")

Link options:
![Front page, link options:](/screenshots/Page1-link-chat-options.png?raw=true "Selecting chat linking options")

Linking saved chat:
![Front page, linking saved chat:](/screenshots/Page1-chat-linked.png?raw=true "Linking chat")

Second page, without selecting a prompt:
![Second page, default](/screenshots/Page2-interaction-default.png?raw=true "Default settings")

Second page, interacting with the model using the chosen prompt and parameters:
![Second page, interaction](/screenshots/Page2-interaction.png?raw=true "Interaction page")

Uploading your own files for additional context:
![Second page, interaction](/screenshots/Page2-interaction-file-upload.png?raw=true "File upload")

Searching the web:
![Second page, interaction](/screenshots/Page2-interaction-web-search.png?raw=true "Web search")

Saving chat:
![Second page, interaction](/screenshots/Page2-interaction-save-chat.png?raw=true "Save chat")

## Usage

### Front Page

#### Selecting a prompt:
The front page offers a comprehensive list of loaded prompts for the user's perusal. Upon entering a query, the system retrieves the four most relevant prompts based on the vector embeddings stored in the Pinecone vector database. The user can then select a prompt, which is subsequently edited by a separate language model (LLM) to remove any default template input. This ensures consistency across all prompts in terms of input expectations.

#### Choosing a model:
Users have the flexibility to choose between gpt-3.5-turbo or gpt-4 models for generating responses. If a user navigates to the "interact-with-llm" page without selecting a prompt, a default interaction with the chosen model will ensue, akin to using chatGPT.

#### Linking chat:
For enhanced user experience, the system allows linking a previous chat with a new one. To execute this, the user simply selects a saved chat they wish to link from the selectbox located underneath the saved chats table on the sidebar. The system then automatically links the selected chat, providing the user with additional options to determine the relevance threshold for messages to be added as context to the new conversation.

#### Show Details:
To provide transparency and insight into the system's operations, the "Show details" checkbox on the sidebar allows users to view detailed information on prompt editing and the rationale behind settings choices.

### Interaction page
#### Normal Interaction:
Once the prompt selection and editing phases are completed, the user can transition to the "interact-with-llm" page. Here, they are greeted with the first response generated by the model using the selected prompt and settings to answer the user's initial query. The user can then interact with the model just as they would ChatGPT - enter a message and click send and the model will generate a response.

#### Using Agent tools:
The "interact-with-llm" page provides users with the ability to upload documents that the model can use as a knowledge base. This feature, accessible via the "Upload files!" checkbox on the sidebar, accepts .pdf, .html, .csv, .txt files. The uploaded document is vectorized and used to answer questions relevant to its contents.

Additionally, the user can choose to provide the model with access to web searching by clicking the "Web Search!" checkbox on the sidebar. This will provide the model with the ability to find up to date information from the internet itself. 

If necessary, users can disable the agent using the "Disable" button on the sidebar, resulting in the main model directly answering any further requests. This can be useful if you no longer wish for the model to search the internet or the documents you provided when answering your question.

#### Saving chat:
The system provides an option to save chats for future reference. To do this, users simply click the "Save chat" button located beneath the user input submission box and assign a name to the saved chat. The system stores up to five most recent chats, which can be viewed on the front page sidebar.

## Technical details
...
(Models, model diagram etc.)

There are 7 models running in the background (Note: a model is considered distinct if its using a diffirent prompt - not different model type or parameters. There are only two model types: gpt-3.5-turbo and gpt-4). These models are the following:
- Edit prompt model (Edits the selected prompt to remove template input)
- Configure Settings model (Chooses the 4 parameters for the main model)
- Main model (Model that uses the selected prompt and parameters to generate response to user query)
- Summarization model (Summarizes chat history)
- Context Agent (Uses tools to provide Main model with context)
- Rating model (Rates the relevance of a message history chunk to a new conversation)
- Link model (Links relevant messages from past conversation to new model)

In addition, there is a Pinecone Vector Database that stores all the pre-defined prompts. A Weaviate cluster is also used to compute vectorstores of any uploaded documents. And SerpAPI is used for web searching functionality.

System overview:
![System overview](/screenshots/System-overview.png?raw=true "System overview")

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

