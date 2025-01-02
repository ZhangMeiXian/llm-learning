# 基于OpenAI API的模型效果评估

import json
from openai import OpenAI

# Load API key from a JSON file.
# Make sure to replace "sk-..." with your actual API key from https://platform.openai.com/api-keys
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    api_key = config["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)


def run_chatgpt(prompt, client, model="gpt-4-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=123,
    )
    return response.choices[0].message.content


prompt = f"Respond with 'hello world' if you got this message."
run_chatgpt(prompt, client)


json_file = "eval-example-data.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data))


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


for entry in json_data[:5]:
    prompt = (f"Given the input `{format_input(entry)}` "
              f"and correct output `{entry['output']}`, "
              f"score the model response `{entry['model 1 response']}`"
              f" on a scale from 0 to 100, where 100 is the best score. "
              )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model 1 response"])
    print("\nScore:")
    print(">>", run_chatgpt(prompt, client))
    print("\n-------------------------")


from tqdm import tqdm


def generate_model_scores(json_data, json_key, client):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the number only."
        )
        score = run_chatgpt(prompt, client)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores


from pathlib import Path

for model in ("model 1 response", "model 2 response"):

    scores = generate_model_scores(json_data, model, client)
    print(f"\n{model}")
    print(f"Number of scores: {len(scores)} of {len(json_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")

    # Optionally save the scores
    save_path = Path("scores") / f"gpt4-{model.replace(' ', '-')}.json"
    with open(save_path, "w") as file:
        json.dump(scores, file)