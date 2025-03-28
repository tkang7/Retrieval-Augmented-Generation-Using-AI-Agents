from src.phase_1.recipe_extractor import RecipeExtractor
from src.utils.import_export import export_recipes_to_csv, import_recipes_to_list_of_objects
from tqdm import tqdm
import pandas as pd
import os

def main():
    recipe_extractor = RecipeExtractor()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_data_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'betty_crocker_cook_book.txt')
    recipes = []

    with open(input_data_path, "r", encoding="utf-8") as f:
        # lines = [line for i, line in enumerate(f, start=1) if 48 <= i <= 1817]
        lines = [line for i, line in enumerate(f, start=1) if 48 <= i <= 70]
        curr_recipe = ""

        for line in tqdm(lines, desc="Processing lines"):   
            if line.strip() == "":
                continue

            is_recipe = recipe_extractor.is_recipe_start(line)
            if is_recipe:
                # print(f"Line - {line.strip()} is a recipe!")
                if curr_recipe.strip():
                    recipe = recipe_extractor.create_recipe_object(curr_recipe)
                    if recipe:
                        recipes.append(recipe)
                curr_recipe = line.strip()
            else:
                curr_recipe += line

        if curr_recipe.strip():
            recipe = recipe_extractor.create_recipe_object(curr_recipe)
            if recipe:
                recipes.append(recipe)

    recipes = recipe_extractor.cross_check_recipe(recipes)
    export_recipes_to_csv(recipes)

    # now try reading in the file as a list of objects
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'recipes.csv')
    data = import_recipes_to_list_of_objects(data_path)

    print(data)

if __name__ == "__main__":
    main()

    # output_data_path = ""