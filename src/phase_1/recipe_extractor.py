from langchain_ollama import ChatOllama
import json

class RecipeExtractor():
    def __init__(self):
        self.llm = ChatOllama(
            model="mistral:latest",
            temperature=0.0
        )
        print("Loaded Mistral 7B via Ollama using LangChain")

    def is_recipe_start(self, text: str) -> bool:
        prompt = (
            "### Instruction:\n"
            "Determine if the following line is the START of a recipe title in a cookbook.\n"
            "Respond only with YES or NO.\n\n"
            f"### Input:\n{text}\n\n### Response:"
        )
        response = self.llm.invoke(prompt)
        return response.content.strip().upper().startswith("YES")
    
    def create_recipe_object(self, text_block: str) -> dict:
        prompt = (
            "### Instruction:\n"
            "Extract the recipe information from the following text block. "
            "Return a JSON object with the following keys: 'title', 'ingredients', 'instructions', "
            "'notes', and 'serving_size'.\n"
            "- The title should be the name of the recipe and MUST be in uppercase.\n"
            "- Ingredients MUST be in a string with each ingredient on its own line.\n"
            "- Instructions MUST be a string with full preparation steps.\n"
            "- Notes should include any extra details or comments (e.g., italicized comments or serving tips).\n"
            "- Serving size should be a list of 2 integers like [6, 8] where each of the values represent [min_serving_size, max_serving_size], or [0, 0] if unspecified.\n"
            "- DO NOT make up or hallucinate ingredients, notes, or steps that are not present in the input.\n\n"
            f"### Input:\n{text_block.strip()}\n\n### Response:"
        )
        # print("Running create_receipe_object...")
        response = self.llm.invoke(prompt)

        try:
            return json.loads(response.content.strip())
        except json.JSONDecodeError:
            print("⚠️ Failed to parse LLM response, raw output below:")
            print(response.content)
            return {}    

    def cross_check_recipe(self, recipes):
        print("Checking over Recipes...")
        recipes = [recipe for recipe in recipes if len(recipe) == 5] # only keep recipe with all keys present

        for recipe in recipes:
            if not recipe["title"].isupper():
                recipe["title"] = recipe["title"].upper()
            if not isinstance(recipe["ingredients"], list):
                recipe["ingredients"] = recipe["ingredients"].split(",")
            if isinstance(recipe["instructions"], list):
                recipe["instructions"] = ". ".join(recipe["instructions"])
            if recipe["serving_size"][0] > recipe["serving_size"][1]:
                recipe["serving_size"][0], recipe["serving_size"][1] = recipe["serving_size"][0], recipe["serving_size"][1] 

        print("Finished Checking over Recipes!")
        
        return recipes
