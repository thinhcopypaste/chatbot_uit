from ollama import chat
from retriever import Retriever
from smooth_context import smooth_contexts
from data_loader import load_meta_corpus
from typing import List, Dict
from openai import OpenAI

from dotenv import load_dotenv
import os
import sys

load_dotenv()

API_OPENAI_KEY = os.getenv("api_openai_key")

client = OpenAI(api_key=API_OPENAI_KEY)

prompt_template = (
    """###Yêu cầu: Bạn là một trợ lý hỗ trợ về học vụ thông minh của Trường Đại học Công Nghệ Thông Tin, chuyên cung cấp câu trả lời dựa trên thông tin được truy xuất từ hệ thống về học vụ của Trường. Khi nhận được dữ liệu truy xuất từ RAG, hãy:  

    1. Phân tích dữ liệu để trả lời đúng trọng tâm câu hỏi của người dùng. Chỉ trả lời dựa trên dữ liệu được cung cấp, không suy đoán hoặc tạo ra thông tin mới.
    2. Tóm tắt thông tin một cách rõ ràng, ngắn gọn nhưng vẫn đầy đủ ý nghĩa.  
    3. Trả lời với giọng điệu thân thiện và dễ tiếp cận.  
    4. Nếu dữ liệu truy xuất không có thông tin liên quan đến câu hỏi hoặc không có dữ liệu nào được truy xuất, hãy trả lời: "Xin lỗi, tôi không có thông tin phù hợp để trả lời câu hỏi này."  
    5. Nếu câu hỏi không liên quan đến chủ đề học vụ của Trường Đại học Công Nghệ Thông Tin (out domain) hãy giới thiệu lịch sự về lĩnh vực của mình.
    6. Trả lời câu hỏi bằng ngôn ngữ: {language}

    ###Dựa vào một số ngữ cảnh truy xuất được dưới đây nếu bạn thấy nó có liên quan đến câu hỏi thì trả lời câu hỏi ở cuối. {input}
    ###Câu hỏi từ người dùng: {question}
    ###Nếu thấy ngữ cảnh có liên quan đến câu hỏi hãy trả lời chi tiết và đầy đủ dựa trên ngữ cảnh."""
    
)

def get_prompt(question, contexts, language):
    context = "\n\n".join([f"Context [{i+1}]: {x['passage']}" for i, x in enumerate(contexts)])
    input = f"\n\n{context}\n\n"
    prompt = prompt_template.format(
        input=input,
        question=question, 
        language=language
    )
    return prompt


def classify_small_talk(input_sentence, language):
    prompt = f"""
    ### Mục tiêu
    Bạn là một trợ lý ảo chuyên về **tư vấn học vụ** của Trường Đại học Công Nghệ Thông Tin. Nhiệm vụ của bạn là **phân loại** mỗi câu hỏi của người dùng thành hai loại:

    1. **Small talk**: các câu chào hỏi, hỏi thăm, cảm ơn, khen ngợi, hay hỏi thông tin cá nhân… **KHÔNG liên quan** đến học vụ.  
    2. **Domain question**: các câu hỏi **liên quan** trực tiếp đến học vụ (ví dụ: chương trình đào tạo, học phí, tín chỉ, lịch thi, quy định…)

    ### Quy tắc trả lời
    - Nếu là **Domain question**, chỉ trả về **chính xác** từ **"no"** (không thêm bất kỳ ký tự, câu giải thích nào).  
    - Nếu là **Small talk**, không trả “no” mà trả về một thông điệp chào mời ngắn gọn, chuyên nghiệp, thân thiện, giới thiệu về chatbot tư vấn học vụ Trường ĐH CNTT, bằng ngôn ngữ {language}.

    ### Ví dụ minh họa

    User query: "Chào bạn, hôm nay bạn thế nào?"  
    Response: "Xin chào! Mình là chatbot tư vấn học vụ Trường Đại học Công Nghệ Thông Tin—sẵn sàng hỗ trợ bạn với mọi thắc mắc về chương trình đào tạo, học phí và học phần. Hãy cho mình biết câu hỏi của bạn nhé! 😊"

    User query: "Điểm số để miễn Anh Văn 2 là bao nhiêu?"  
    Response: "no"

    User query: "Bạn tên là gì?"  
    Response: "Xin chào! Mình là chatbot tư vấn học vụ Trường Đại học Công Nghệ Thông Tin—sẵn sàng hỗ trợ bạn với mọi thắc mắc về chương trình đào tạo, lịch thi, học phí và học phần. Hãy cho mình biết câu hỏi học vụ của bạn nhé! 😊"

    User query: "Chương trình tiên tiến là gì?"  
    Response: "no"

    User query: "Cảm ơn!"  
    Response: "Cảm ơn bạn đã tin tưởng! Mình là chatbot tư vấn học vụ Trường Đại học Công Nghệ Thông Tin—luôn sẵn sàng giải đáp mọi thắc mắc liên quan đến chương trình đào tạo, tín chỉ và học phần. Hãy hỏi mình bất cứ điều gì về học vụ nhé! 😊"

    ### Thực thi phân loại
    Dựa vào câu hỏi của người dùng, thực hiện đúng quy tắc trên.  
    Câu hỏi từ người dùng: {input_sentence}
    """


    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    answer = completion.choices[0].message.content
    return answer.strip().lower()

def create_new_prompt(prompt, chat_history, user_query, **kwargs):
  new_prompt = f"{prompt} lịch sử cuộc trò chuyện: {chat_history} câu hỏi của người dùng: {user_query}"
  for key, value in kwargs.items():
    new_prompt += f" {key}: {value}"

  return new_prompt

def chatbot(conversation_history: List[Dict[str, str]], language) -> str:
    user_query = conversation_history[-1]['content']

    meta_corpus = load_meta_corpus(r"ChatBotUIT-master\data\corpus_chunks.jsonl")
    # for doc in meta_corpus:
    #     if "passage" not in doc:
    #         doc["passage"] = doc.get("context", "")


    retriever = Retriever(
        corpus=meta_corpus,
        corpus_emb_path=r"ChatBotUIT-master\data\corpus_embedding.pkl",
        model_name="ChatBotUIT-master\model\model_embedding"
    )

    # Xử lý nếu người dùng có câu hỏi nhỏ hoặc trò chuyện phiếm
    result = classify_small_talk(user_query, language)
    print("result classify small talk:", result)
    if "no" not in result:
        return result

    elif "no" in result:
        prompt = """Dựa trên lịch sử cuộc trò chuyện và câu hỏi mới nhất của người dùng, có thể tham chiếu đến ngữ cảnh trong lịch sử trò chuyện, 
            hãy tạo thành một câu hỏi độc lập có thể hiểu được mà không cần lịch sử cuộc trò chuyện. 
            KHÔNG trả lời câu hỏi, chỉ cần điều chỉnh lại nếu cần, nếu không thì giữ nguyên. 
            Nếu câu hỏi bằng tiếng Anh, sau khi tinh chỉnh, hãy dịch câu hỏi đó sang tiếng Việt."""

        # Sử dụng hàm tạo câu hỏi mới từ lịch sử trò chuyện
        new_prompt = create_new_prompt(
            prompt=prompt,
            chat_history=conversation_history,
            user_query=user_query,
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": new_prompt}
            ]
        )

        answer = completion.choices[0].message.content
        print("Câu hỏi mới: ", answer)
        question = answer
        top_passages = retriever.retrieve(question, topk=10)
        for doc in top_passages:
            if "passage" not in doc:
                doc["passage"] = doc.get("context", "")

        print("topK:", top_passages)
        smoothed_contexts = smooth_contexts(top_passages, meta_corpus)
        print("Smooth context: ", smoothed_contexts)
        prompt = get_prompt(question, smoothed_contexts, language)
    
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = completion.choices[0].message.content
        
        return answer

    else:
        print("Unexpected response from the model.")
        return "Xin lỗi, hệ thống không xử lý được."
    
# def main():
#     # Nhận input từ người dùng
#     user_query = input("User query: ")

#     result = chatbot(user_query)

#     # Trả về output
#     print(result)

# if __name__ == "__main__":
#     main()
