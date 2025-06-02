# gcp_vertexai_demo.py

from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="smart-impact-459512-f3",
      location="global",
  )


  model = "gemini-2.5-flash-preview-05-20"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text="""I want to compare hyperscalers to see which is the best option for enterprise?""")
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
      threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
      threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
      threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold=types.HarmBlockThreshold.BLOCK_NONE
    )],
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()