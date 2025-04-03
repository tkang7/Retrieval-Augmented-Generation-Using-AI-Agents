import re
import numpy as np
from phase_2.vector_store import VectorStore

def normalize(text):
  return re.sub(r'\s+', ' ', text.strip())

k = 3
vs = VectorStore()
vs.create()
vector_store = vs.vector_store
embedder = vs.embedder


test_cases = [('TUNA BROCCOLI CASSEROLE notes', 0, 'notes', 'Broccoli right in your biscuits!'),
              ('Fruit Short Pie ingredients', 0, 'ingredients', '2 tbsp. Bisquick\n1 cup sugar\n½ tsp. cinnamon\n1 cup water\n1 tbsp. lemon juice\n4 cups fresh blueberries, peaches, or cherries'),
              ('watermelon', 0, 'text', 'No matching documents!'),
              ('SWEDISH PANCAKES instructions', 0, 'instructions', 'Beat together until blended. Lightly grease a 6 or 7″ skillet. Spoon about 3 tbsp. batter into hot skillet and tilt to coat bottom of pan. Cook until small bubbles appear on surface. Loosen edges with spatula, turn pancake gently and finish baking on other side. Lay on towel or absorbent paper; place in low oven to keep warm. Spread each with sugar, jam, applesauce, or whipped cream, etc. and roll up like jelly roll. Serve warm.'),
              ('DEVILED HAM TURNOVERS', 0, 'text', 'DEVILED HAM TURNOVERS\nHeat oven to 450° (hot). Make Biscuit or Fruit Shortcake dough (p. 3). Roll into 15″ square on surface lightly dusted with Bisquick. Cut into twenty-five 3″ squares. Place on ungreased baking sheet. Spoon a little Ham Filling onto center of each square. Make triangle by folding one half over the other so top edge slightly overlaps. Press edges together with a fork dipped in cold water. Bake 8 to 10 min. Ham Filling: Blend two 2¼-oz. cans deviled ham and 2 tbsp. cream.'),
              ('cranberry muffin ingredients', 0, 'ingredients', '¾ cup raw cranberries (cut in halves or quarters) ½ cup confectioners’ sugar'),
              ('vanilla pudding', 0, 'text', 'No matching documents!'),
              ('strawberry pie', 0, 'text', 'STRAWBERRY GLACÉ SHORT PIE\n1 qt. strawberries\n1 cup water\n1 cup sugar\n3 tbsp. cornstarch\nWash, drain, and hull strawberries. For glaze, simmer 1 cup of the berries with ⅔ cup water until berries start to break up (about 3 min.). Blend sugar, cornstarch, remaining ⅓ cup water; stir into boiling mixture. Boil 1 min., stirring constantly. Cool. Pour remaining 3 cups of berries into baked Short Pie (p. 8). Cover with glaze. Refrigerate until firm ... about 2 hr. Top with whipped cream or ice cream.'),
              ('coffee cake', 1, 'text', 'BANANA COFFEE CAKE Make Coffee Cake batter (p. 2)—except add 1 cup mashed, fully ripe bananas in place of milk. Bake.'),
              ('noodles notes', 0, 'notes', 'A new easier way to make real homemade noodles.'),
              ('helicopter ingredients', 0, 'text', 'No matching documents!'),
              ('WHUFFINS', 0, 'instructions', 'WHUFFINS Make richer Muffins (p. 2)—except fold 1½ cups Wheaties carefully into batter.'),
              ('asparagus cake serving size', 0, 'serving size', [6,6]),
              ('cinnamon doughnuts ingredients', 0, 'ingredients', '2 cups Bisquick ¼ cup sugar ⅓ cup milk 1 tsp. vanilla 1 egg ¼ tsp. each cinnamon and nutmeg, if desired'),
              ('pineapple buns', 0, 'text', 'PINEAPPLE STICKY BUNS ¾ cup drained crushed pineapple ½ cup soft butter ½ cup brown sugar (packed) 1 tsp. cinnamon Heat oven to 425° (hot). Mix ingredients and divide among 12 large greased muffin cups. Make Fruit Shortcake dough (p. 3). Spoon over pineapple mixture. Bake 15 to 20 min. Invert on tray or rack immediately to prevent sticking to pans.' )]

values = np.arange(0, 1.05, 0.05)
best_score = float('-inf')
best_min_similarity = None

for val in values:
  score = 0
  for (query, k_index, detail, expected_result) in test_cases:
    results = vs.search(embedder, vector_store, query, k=k, min_similarity=val)
    try:
      result = normalize(results[k_index])
      expected_result = normalize(expected_result)
      if result == expected_result:
        score += 1
    except AttributeError:
      if results[k_index] == expected_result:
        score += 1
    except IndexError:
        result_str = 'results' if k_index > 1 else 'result'
  print(f"For min similarity {val}, score --> {score}")
  
  if score > best_score:
    best_score = score
    best_min_similarity = val

print(f"\nBEST MIN SIMILARITY IS {best_min_similarity}")

for (query, k_index, detail, expected_result) in test_cases:
  results = vs.search(embedder, vector_store, query, k=k, min_similarity=best_min_similarity)
  print(f'Test case: "{query}"')
  try:
    result = normalize(results[k_index])
    expected_result = normalize(expected_result)
    if result == expected_result:
      score += 1
      print('PASSED')
    else:
      print(f'FAILED: returned incorrect {detail}')
      print('Returned: ', result)
      print('Expected: ', expected_result)
  except AttributeError:
    if results[k_index] == expected_result:
      score += 1
      print('PASSED')
    else:
      print(f'FAILED: returned incorrect serving size')
      print('Returned: ', results[k_index])
      print('Expected: ', expected_result)
  except IndexError:
      result_str = 'results' if k_index > 1 else 'result'
      print(f'FAILED: the query "{query}" returned less than {k_index+1} {result_str}')
  print('------')


print(f'Score: {score}/{len(test_cases)}')