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
    
    def __clean_llm_json(raw: str):
        # Remove lines starting with // or **
        cleaned = re.sub(r"^\s*(//|\*\*).*?$", "", raw, flags=re.MULTILINE)
        # Remove trailing commas before ] or }
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
        return cleaned.strip()

    def create_recipe_object(self, text_block: str) -> dict:
        prompt = (
            "### Instruction:\n"
            "Extract the recipe information from the following text block. "
            "Return a valid JSON object with the following exact keys: 'title', 'ingredients', 'instructions', 'notes', and 'serving_size'.\n"
            "- 'title' MUST be in uppercase and reflect the actual recipe title.\n"
            "- 'ingredients' MUST be a single string with each ingredient on a new line. If no ingredients are present, return an empty string.\n"
            "- 'instructions' MUST be a complete string of preparation steps. Do not hallucinate or add missing steps.\n"
            "- 'notes' MUST be based only on explicit text present (e.g., italicized lines). If none, use an empty string.\n"
            "- 'serving_size' MUST be a list of two integers like [4, 6]. Use [0, 0] if unspecified.\n"
            "- DO NOT add comments (// or **) or markdown formatting.\n"
            "- DO NOT include any text before or after the JSON object.\n"
            "- DO NOT include trailing commas.\n"
            "- Ensure all list items (like in ‘ingredients’) are separated by commas."
            "- The response MUST be a valid JSON object, parsable with json.loads().\n\n"
            f"### Input:\n{text_block.strip()}\n\n### Response:"
        )
        # print("Running create_receipe_object...")
        response = self.llm.invoke(prompt)

        response = self.clean_llm_json(response.content)

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
