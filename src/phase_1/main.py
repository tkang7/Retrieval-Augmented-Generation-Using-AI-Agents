from recipe_extractor import RecipeExtractor
from tqdm import tqdm
import pandas as pd

def export_recipes_to_csv(recipes):
    df = pd.DataFrame(recipes)
    df.to_csv("../../data/processed/recipes.csv", index=False)

def main():
    recipe_extractor = RecipeExtractor()
    recipes = []

    with open("../../data/raw/betty_crocker_cook_book.txt", "r", encoding="utf-8") as f:
        # lines = [line for i, line in enumerate(f, start=1) if 48 <= i <= 1817]
        lines = [line for i, line in enumerate(f, start=1) if 48 <= i <= 100]
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

if __name__ == "__main__":
    main()