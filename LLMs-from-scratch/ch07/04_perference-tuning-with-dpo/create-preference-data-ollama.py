# 基于ollama创建偏好学习数据集
# 启动ollama：ollama serve
# 运行llama3：ollama run llama3
# 加载模型
import urllib.request
import json


def query_model(prompt, model="llama3.1:70b", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
        }
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


result = query_model("What do Llamas eat?")
print(result)


from pathlib import Path

json_file = Path("..", "01_main-chapter-code", "instruction-data.json")

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


import random

for entry in json_data[:5]:
    politeness = random.choice(["polite", "impolite"])
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"slightly rewrite the output to be more {politeness}."
        "Keep the modification minimal."
        "Only return return the generated response and nothing else."
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print(f"\n{politeness} response:")
    print(">>", query_model(prompt))

import random
from tqdm import tqdm


def generate_model_responses(json_data):
    for i, entry in enumerate(tqdm(json_data, desc="Writing entries")):
        politeness = random.choice(["polite", "impolite"])
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"slightly rewrite the output to be more {politeness}."
            "Keep the modification minimal."
            "Only return return the generated response and nothing else."
        )
        response = query_model(prompt)

        if politeness == "polite":
            json_data[i]["chosen"] = response
            json_data[i]["rejected"] = entry["output"]
        else:
            json_data[i]["rejected"] = response
            json_data[i]["chosen"] = entry["output"]


generate_model_responses(json_data)


with open("instruction-data-with-preference.json", "w") as file:
    json.dump(json_data, file, indent=4)