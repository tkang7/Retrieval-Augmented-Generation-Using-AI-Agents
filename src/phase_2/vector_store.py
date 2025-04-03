import json
import ast
from phase_2.embedder import Embedder
from sklearn.metrics.pairwise import cosine_similarity
import os

class VectorStore:
    def __init__(self):
        self.embedder = Embedder().create()
        self.vector_store = None

        this_dir = os.path.dirname(__file__)
        data_path = os.path.join(this_dir, '../../data/processed/recipes.json')
        with open(os.path.abspath(data_path)) as f:
            self.recipe_data = json.load(f)
        
    
    def create(self):
        self.vector_store = list() #don't change the name of this variable! It's required for the testing code

        #Step 1 - buildvector store
        for recipe in self.recipe_data:
            text = "\n".join(
                part for part in [recipe.get('title'), recipe.get('ingredients'), recipe.get('instructions')]
                if part is not None
            )
            embeddings = [self.embedder.encode(text)]
            metadata = recipe

            self.vector_store.append(dict({
                "text": text,
                "embedding": embeddings,
                "metadata": metadata
            }))
    
    def get(self):
        if self.vector_store:
            return self.vector_store
        else:
            print(f"Vector Store not created. Create on by calling create_vector_store first.")
            return None
    
    def search(self, embedder, vector_store, query, k, min_similarity):
        query_split = query.split()
        ingredient_keyword = 'ingredients'
        instructions_keyword = 'instructions'
        note_keyword = 'notes'
        serving_size = ['serving', 'size']

        if ingredient_keyword in query_split:
            keyword = ingredient_keyword
        elif instructions_keyword in query_split:
            keyword = instructions_keyword
        elif note_keyword in query_split:
            keyword = note_keyword
        elif any(word in serving_size for word in query_split):
            keyword = "_".join(serving_size)
            query_split.remove("serving")
            query_split.remove("size")
        else:
            keyword = ''
        
        if keyword and keyword != "_".join(serving_size): 
            query_split.remove(keyword)
        search_text = " ".join(query_split).strip()

        target_text_enc = [embedder.encode(search_text)]
        all_sim = []

        for vector in vector_store:
            curr_text_enc = vector['embedding']
            sim = cosine_similarity(target_text_enc, curr_text_enc)[0][0]
            if sim >= min_similarity:
                all_sim.append([vector, sim])

        top_k = sorted(all_sim, key=lambda x: x[1], reverse=True)[:k]
        search_result = []
        
        if not keyword:
            for data in top_k:
                recipe_metadata = data[0]['text']
                if recipe_metadata:
                    search_result.append(recipe_metadata)
        else:
            for data in top_k:
                recipe_keyword_data = data[0]['metadata'][keyword]

                if recipe_keyword_data and keyword == "_".join(serving_size):
                    search_result.append(ast.literal_eval(recipe_keyword_data))
                elif recipe_keyword_data: 
                    search_result.append(recipe_keyword_data)
        
        if search_result: 
            return search_result
        else:
            return ["No matching documents!"]

if __name__ == "__main__":
    vs = VectorStore()
    vs.create()
    vector_store = vs.vector_store
    embedder = vs.embedder

    k = 1
    min_similarity = 0.2
    query = "cranberry muffin ingredients"
    results = vs.search(embedder, vector_store, query, k, min_similarity)
    print(results)