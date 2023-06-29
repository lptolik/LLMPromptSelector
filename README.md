# Prompt Selector Bot

## Name
Prompt selector.

## Description
Main functionality: selects a "professional" prompt from a list of pre-defined prompts for the user to use based on their input. Additionally selects the best suited parameters for the chosen prompt to get the best results. Saves the average user time writing a prompt from scratch/finding a prompt from a large list of existing ones as well as selecting the most appropriate model parameters. Under the assumption that the average user is not familiar with "Prompt engineering techniques" or what makes a good prompt, this program automates prompt selection and configuration.

Since the prompt is selected from existing written ones stored in "prompts_data_updated.csv", potentially nonsensical llm-generated prompts are avoided.

Additional functionality: Context upload.
Context, in the form a .pdf, .html, .csv, .txt file can be uploaded, which will then be used as a knowledge base to respond to user queries. This feature combats hallucinations by allowing the user to provide their own trusted information.

## Visuals
Front page, prompt selection:
![Front page, prompt selection:](/screenshots/Page1-upon-load.png?raw=true "Front page")

![Front page, prompt selection:](/screenshots/Page1-upon-prompt-selection.png?raw=true "Front page")

![Front page, prompt configuring:](/screenshots/Page1-upon-prompt-configuring-no-details.png?raw=true "Cofiguring prompt")

![Front page, prompt configuration with details:](/screenshots/Page1-upon-prompt-configuring-yes-details.png?raw=true "Congifuring prompt and showing reasoning")

Second page, without selecting a prompt:
![Second page, default](/screenshots/Page2-interaction-default.png?raw=true "Default settings")

Second page, interacting with the model using the chosen prompt and parameters:
![Second page, interaction](/screenshots/Page2-interaction.png?raw=true "Interaction page")

![Second page, interaction](/screenshots/Page2-interaction-file-upload.png?raw=true "File upload")

## Usage
On the front page, the full list of loaded prompts is always available to view. Model selection between gpt-3.5-turbo or gpt-4 is available to change the model used to generate responses. Switching to the "interact-with-llm" page without first selecting a prompt will result in a default interaction with the chosen model similar to using chatGPT.

Entering your query on the front page will result in retrieval of the 4 most relevant prompts you can choose to use, according to the vector embeddings stored using the Pinecone vector database. Upon selecting a prompt, the prompt will be edited to remove template input that some prompts have. This editing phase, conducted by a separate llm, ensures all prompts are consistent in their expectations of input. Another llm then chooses the most suitable parameters based on the edited prompt. To view details on prompt editing and settings choice reasoning, the user can select the "Show details" checkbox located on the sidebar. Finally, once both phases are complete, the user can move onto the "interact-with-llm" page where they will be greeted with the first response produced by the configured model using the chosen prompt to answer the users initial query.

To upload a document the model will use as a knowledge base, the "interact-with-llm" page has a "Upload files!" checkbox located on the sidebar. Upon selection, the user can upload a .pdf, .html, .csv, .txt file and provide a name and description of when to use. This document will be vectorised and used as a tool to answer questions relevant to the contents of the knowledge base. Multiple files can be uploaded, which would result in multiple tools being available to answer your questions.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
