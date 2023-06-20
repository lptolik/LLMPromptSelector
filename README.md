# Prompt Selector Bot

## Name
Prompt selector.

## Description
Main functionality: selects an appropriate "professional" prompt for the user to use based on their input. Additionally selects the best suited parameters for an appropriate answer. Saves the average user time writing a prompt from scratch/finding a prompt from a large list of existing ones as well as selecting the most appropriate model parameters. Under the assumption that the average user is not familiar with "Prompt engineering techniques" or what makes a good prompt, this program automates prompt selection and configuration.

Since the prompt is selected from existing written ones stored in "prompts_data_updated.csv", potentially nonsensical llm-generated prompts are avoided. 

## Visuals
Front page, prompt selection:
![Front page, prompt selection:](/screenshots/Page1-upon-load.png?raw=true "Front page")

![Front page, prompt selection:](/screenshots/Page1-upon-prompt-selection.png?raw=true "Front page")

Second page, interacting with the model using the chosen prompt and parameters:
![Second page, interaction](/screenshots/Page2-interaction.png?raw=true "Interaction page")

## Usage
On the front page, the full list of loaded prompts is always available to view. Model selection between gpt-3.5-turbo or gpt-4 is available to change the model used to generate responses. Switching to the "interact-with-llm" page without first selecting a prompt will result in a default interaction with the chosen model similar to using chatGPT.

Entering your query on the front page will result in retrieval of the 4 most relevant prompts you can choose to use, according to the vector embeddings stored using the Pinecone vector database. Upon selecting a prompt, the prompt will be edited to remove template input that some prompts stored have. This editing phase, conducted by a separate llm, ensures all prompts are consistent on their expectations for input. Another llm then decides on the most suitable parameters based on the edited prompt. Finally, once both phases are complete, the user can move onto the "interact-with-llm" page where they will be greeted with the first response produced by the configured model using the chosen prompt to the users initial query.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
