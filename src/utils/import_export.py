import pandas as pd
import os

def export_recipes_to_csv(recipes):
    print("Exporting to CSV...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'recipes.csv')

    df = pd.DataFrame(recipes)
    df.to_csv(path, index=False)

    print("Finished Exporting to CSV...")

def import_recipes_to_list_of_objects(csv_path):
    print("Exporting to List of Objects...")

    df = pd.read_csv(csv_path)
    data = df.to_dict(orient='records')

    print("Finished Exporting to List of Objects...")
    
    return data
    
    