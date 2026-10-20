
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["REQUESTY_API_KEY"],
    base_url="https://router.requesty.ai/v1",
)

response = client.chat.completions.create(
    model="deepinfra/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    messages=[
        {"role": "user", "content": "Explain what ECLASS is in three sentences."}
    ],
)

print(response.choices[0].message.content)
