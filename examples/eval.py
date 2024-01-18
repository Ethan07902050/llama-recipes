import json
import evaluate
from tqdm import tqdm

pred_path = '/work/u1509343/storydalle/data/pororo_png/background_llama_test_10_epoch.json'
ref_path = '/work/u1509343/storydalle/data/pororo_png/background_prompt_test.json'

pred_dict = json.load(open(pred_path))
ref_dict = json.load(open(ref_path))
keys = ref_dict.keys()

predictions, references = [], []
for key in tqdm(keys):
    predictions.append('\n'.join(pred_dict[key]))
    references.append('\n'.join(ref_dict[key]))

rouge = evaluate.load('rouge')
results = rouge.compute(predictions=predictions, references=references)
print(results)