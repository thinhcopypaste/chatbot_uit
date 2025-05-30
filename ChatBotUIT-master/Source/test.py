# main.py

from data_loader import load_meta_corpus
from retriever import Retriever
from smooth_context import smooth_contexts
from chat import chatbot

# Load corpus
meta_corpus = load_meta_corpus("D:\ChatBotMonNhanDang\data\corpus_chunks.jsonl")
# for doc in meta_corpus:
#     if "passage" not in doc:
#         doc["passage"] = doc.get("context", "")

# Initialize retriever
retriever = Retriever(
    corpus=meta_corpus,
    corpus_emb_path="D:\ChatBotMonNhanDang\data\corpus_embedding.pkl",
    model_name="D:\ChatBotMonNhanDang\model\model_embedding"
)

# Query example
question = "Địa điểm du lịch nổi tiếng ở miền Bắc là gì?"
top_results = retriever.retrieve(question, topk=10)
for doc in top_results:
    if "passage" not in doc:
        doc["passage"] = doc.get("context", "")

# Smooth contexts
smoothed_contexts = smooth_contexts(top_results, meta_corpus)

# Display results
# for context in smoothed_contexts:
#     print(context["passage"], context["score"])
# {"role": "user", "content": "Địa điểm du lịch ở An Giang?"},
#                 {"role": "system", "content": "An Giang có nhiều điểm du lịch thú vị."},
# Chatbot interaction
conversation_history = [
                {"role": "user", "content": "Chào bạn, bạn thích đi du lịch chứ?"}]
response = chatbot(conversation_history, "Tiếng Việt")
print(response)