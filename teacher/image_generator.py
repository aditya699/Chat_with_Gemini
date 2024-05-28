'''
This will help teachers to generate images using DALLE-2
'''
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
import requests
import os
from openai import AzureOpenAI
import json
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"]=os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"]=os.getenv("AZURE_OPENAI_API_KEY")
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
n=input("Enter the topic on which you which to generate image on : ")
result = client.images.generate(
    model="sass", # the name of your DALL-E 3 deployment
    prompt=f"{n}",
    n=1
)

image_url = json.loads(result.model_dump_json())['data'][0]['url']

print(image_url)

import getpass
os.environ["STABILITY_KEY"]=os.getenv("STABILITY_KEY")
# @markdown To get your API key visit https://platform.stability.ai/account/keys
STABILITY_KEY = os.environ["STABILITY_KEY"]

def send_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response


prompt = n #@param {type:"string"}
negative_prompt = "" #@param {type:"string"}
aspect_ratio = "1:1" #@param ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
seed = 0 #@param {type:"integer"}
output_format = "jpeg" #@param ["jpeg", "png"]

host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"

params = {
    "prompt" : prompt,
    "negative_prompt" : negative_prompt,
    "aspect_ratio" : aspect_ratio,
    "seed" : seed,
    "output_format" : output_format,
    "model" : "sd3",
    "mode" : "text-to-image"
}

response = send_generation_request(
    host,
    params
)

# Decode response
output_image = response.content
finish_reason = response.headers.get("finish-reason")
seed = response.headers.get("seed")

# Check for NSFW classification
if finish_reason == 'CONTENT_FILTERED':
    raise Warning("Generation failed NSFW classifier")

# Save and display result
generated = f"generated_{seed}.{output_format}"
with open(generated, "wb") as f:
    f.write(output_image)
print(f"Saved image {generated}")