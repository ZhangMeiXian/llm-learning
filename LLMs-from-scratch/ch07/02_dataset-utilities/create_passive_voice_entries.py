# 允许样本中添加"passive voice"，更符合日常对话表达
from importlib.metadata import version

pkgs = ["openai",  # OpenAI API
        "tqdm",    # Progress bar
       ]

for p in pkgs:
    print(f"{p} version: {version(p)}")


# 测试OpenAI
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
    )
    return response.choices[0].message.content


# Prepare input
sentence = "I ate breakfast"
prompt = f"Convert the following sentence to passive voice: '{sentence}'"
run_chatgpt(prompt, client)

import json

json_file = "instruction-examples.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data)) # 100

for entry in json_data[:5]:
    text = entry["output"]
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"

    print("\nInput:")
    print(">>", text)
    print("\nOutput:")
    print(">>", run_chatgpt(prompt, client))
    print("\n-------------------------")


from tqdm import tqdm  # a progress bar tool


for i, entry in tqdm(enumerate(json_data[:5]), total=len(json_data[:5])):
    text = entry["output"]
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"
    json_data[i]["output_2"] = run_chatgpt(prompt, client)


# 用GPT转换成passive voice语态
for i, entry in tqdm(enumerate(json_data), total=len(json_data)):
    text = entry["output"]
    prompt = f"Without adding any response or explanation, convert the following text to passive voice: {text}"
    json_data[i]["output_2"] = run_chatgpt(prompt, client)


new_json_file = json_file.replace(".json", "-modified.json")


with open(new_json_file, "w") as file:
    json.dump(json_data, file, indent=4)  # "indent" for pretty-printing