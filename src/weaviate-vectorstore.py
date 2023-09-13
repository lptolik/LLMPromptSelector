import weaviate
import pandas as pd
import json

class vectorstore():
    client: weaviate.client

    def __init__(self):
        client = weaviate.Client(
            url = "https://prompt-cluster-8z6hr799.weaviate.network",  # Replace with your endpoint
            additional_headers = {
                "X-OpenAI-Api-Key": ""  # Replace with your inference API key
            }
        )

        class_obj = {
            "class": "Prompts",
            "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",  # Can be any public or private Hugging Face model.
                    "modelVersion": "002",
                    "type": "text"
                }
            }
        }

        client.schema.create_class(class_obj)

    def load_data(self):
        data = pd.read_csv("src/prompt_data_updated.csv").to_list()

        # Configure a batch process
        with self.client.batch(
            batch_size=100
        ) as batch:
            # Batch import all Questions
            for i, d in enumerate(data):
                print(f"importing prompt: {i+1}")

                properties = {
                    "prompt_text": d,
                }

                self.client.batch.add_data_object(
                    properties,
                    "Prompts",
                )

# Now we add the prompts to the database using a batch process:

# Load data
