import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import Counter
import os, time
import json
import argparse
from tqdm import tqdm
from au_eu_utils import get_data, uncertainty_calculation, token_uncertainty_calculation_new, answer_extraction

parser = argparse.ArgumentParser()

import warnings

warnings.filterwarnings('ignore')


PROMPT_TEMPLATE_1 = f"""
Classify the following sentence into six categories: [0: Sadness; 1: Joy, 2: Love; 3: Anger; 4: Fear, 5: Surprise].
Provide answer in a structured format WITHOUT additional comments, I just want the numerical label for each sentence.
"""


def main(model, tokenizer, prompts, training_data, args):
    path = os.path.join(args.save_path, str(int(args.current_time)))
    # Create new dir
    if not os.path.exists(path):
        os.makedirs(path)
    # Save configrations
    with open(path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Append to the saved path
    data = []
    for index, prompt in tqdm(enumerate(prompts)):
        preds, entropies = uncertainty_calculation(model, tokenizer, prompt, training_data,
                                                   args.decoding_strategy, args.num_demos,
                                                   args.num_demos_per_class, args.sampling_strategy, 
                                                   args.iter_demos)

        AU, EU = token_uncertainty_calculation_new(preds, entropies, num_classes=6)
        print("AU: {}\tEU: {}\tAU_new: \tEU_new: ".format(AU, EU))
        pred = answer_extraction(preds)
        try:
            pred = Counter(pred).most_common()[0][0]
        except:
            pred = None
        save_res = {"Question": prompt, "Label": labels[index], "Predicted_Label": pred, "AU": AU, "EU": EU}
        
        data.append(save_res)
    return data


def format_json(df):
    result = df[['AU', 'EU']].to_dict(orient='records')
    for i, record in enumerate(result):
        record['Id'] = i
        record['Entropy'] = record['AU'] + record['EU']
    return result


if __name__ == '__main__':
    parser.add_argument('--save_path', type=str, default='results/')
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--file', type=str, default="xxx.json", help="Specify which dataset to use")
    parser.add_argument('--model', type=str, default='7b')
    parser.add_argument('--num_demos', type=int, default=4)
    parser.add_argument('--num_demos_per_class', type=int, default=1)
    parser.add_argument('--sampling_strategy', choices=['random', 'class'], default='class')
    parser.add_argument('--decoding_strategy', choices=['beam_search', 'constractive', 'greedy', 'top_p'],
                        default='beam_search')
    parser.add_argument('--iter_demos', type=int, default=4)
    parser.add_argument('--load8bits', default=False, help='load model with 8 bits')
    parser.add_argument('--current_time', type=str, default=time.time())
    parser.add_argument('--num_data', type=int, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Model
    model_path = '{}'.format(args.model)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", fp16=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path,  device_map="auto", trust_remote_code=True)
    print("Done! Loaded Model: {}".format(args.model))

    # Loading Data
    training_data, test_data = get_data(f"{args.data_path}/{args.file}")
    
    prompts = [i for i in test_data]
    labels = [i['answer'] for i in test_data]
    print("Done! Loaded Data")

    # perform AU/EU Decomposition
    data = pd.DataFrame(main(model, tokenizer, prompts[:args.num_data], training_data, args))
    
    # format to output
    data = pd.DataFrame(data)
    filepath = 'au-eu-outputs/{}_{}_au_eu.json'
    with open(filepath.format(
        args.model.split('/')[-1],
        args.file.split('.')[0],
    ), 'w') as f:
        json.dump(format_json(data), f)
