# Prompt Selector Bot

## Name
Prompt selector.

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
![Front page, Selecting saved chat:](/screenshots/Page1-choosing-saved-chat.png?raw=true "Selecting saved chat")

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

# Basic front page usage:
On the front page, the full list of loaded prompts is always available to view. Model selection between gpt-3.5-turbo or gpt-4 is available to change the model used to generate responses. Switching to the "interact-with-llm" page without first selecting a prompt will result in a default interaction with the chosen model similar to using chatGPT.

Entering your query on the front page will result in retrieval of the 4 most relevant prompts you can choose to use, according to the vector embeddings stored using the Pinecone vector database. Upon selecting a prompt, the prompt will be edited to remove default template input that some prompts have. This editing phase, conducted by a separate llm, ensures all prompts are consistent in their expectations of input. Another llm then chooses the most suitable parameters based on the edited prompt. To view details on prompt editing and settings choice reasoning, the user can select the "Show details" checkbox located on the sidebar. Finally, once both phases are complete, the user can move onto the "interact-with-llm" page where they will be greeted with the first response produced by a model using the chosen prompt and the chosen settings to answer the users initial query.

# Second page usage:
To upload a document the model will use as a knowledge base, the "interact-with-llm" page has a "Upload files!" checkbox located on the sidebar. Upon selection, the user can upload a .pdf, .html, .csv, .txt file and provide a name and description of when to use. This document will be vectorised and used as a tool to answer questions relevant to the contents of the knowledge base. Multiple files can be uploaded, which would result in multiple tools being available to answer your questions. Any user query will first be serviced by an Agent using the generated tools, the findings of the Agent will then be passed back to the main model which will respond using the found context and the selected prompt.

To search the web, the "interact-with-llm" page has a "Web search!" checkbox located on the sidebar. Upon selection, any user query will first be serviced by an agent using the web search, before providing found context to the main model which will respond to the user using the found context in the style of the prompt selected.

On occasion, the user may want to disable the agent so any further requests are answered directly by the main model. This is helpful if a user query is not asking for any additonal context that could be found in an Agent tool. If an Agent is enabled, a "Disable" button is available on the sidebar, which will turn of the Agent.

To save a chat, simply click the "Save chat" button underneath the user input submission box and provide a name to the saved chat.

# Additional front page features:
If a chat has been saved, it will appear on the front page on the sidebar where the user may view it. Up to 5 most recent chats can be saved. 

The user also has the option of linking a previous chat with a new one. To do this, simply select any saved chat you would like to link in the selectbox underneath the saved chats table on the sidebar. Now, if the user chooses a prompt, the program will automatically proceed to link the saved chat. The user then has further options to restrict how relevant a message must be to be added as context to the new conversation using a slidebar.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
