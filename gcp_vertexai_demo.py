# gcp_vertexai_demo.py

from vertexai.language_models import ChatModel
import vertexai
import time

vertexai.init(project="<your-project-id>", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison")
chat = chat_model.start_chat()

start = time.time()
response = chat.send_message("Explain Retrieval-Augmented Generation (RAG).")
end = time.time()

print(f"GCP Chat Bison Response (Latency: {round(end - start, 2)}s):\n", response.text)