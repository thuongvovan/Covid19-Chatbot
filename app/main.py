from flask import Flask, render_template, request
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize

# ---------------- PHẦN CHATBOT _________________________
# Đọc dữ liệu mẫu
data= pd.read_excel('data.xlsx')
stop_words= open("stopwords.words", "r").read()
stop_words= stop_words.splitlines()

# Chuẩn hóa ký tự trong dữ liệu huấn luyện
def standardized_sentence(sentence):
    sentence= word_tokenize(sentence.lower(), format="text")
    return sentence

sample_questions= data.QUESTION
sample_questions= sample_questions.apply(standardized_sentence)
sample_questions= list(sample_questions)

sample_answer= list(data.ANSWER)

# Mô hình dự đoán
tfidf_vectorizer = TfidfVectorizer(stop_words= None).fit(sample_questions)
tfidf = tfidf_vectorizer.transform(sample_questions)

# Mẫu xin lỗi
sorry_response= ['Xin lỗi, Tôi chưa hiểu câu hỏi của bạn hoặc có thể vấn đề bạn hỏi nằm ngoài sự hiểu biết của tôi.',
                 'Xin lỗi, Tôi chưa hiểu câu hỏi của bạn.',
                 'Xin lỗi, có thể vấn đề bạn hỏi nằm ngoài sự hiểu biết của tôi.',
                 'Bạn có thể nhắc lại câu hỏi lần nữa được không?']

# Các mẫu chào hỏi
greeting_inputs= ('xin chào', 'chào bạn', 'hello', 'hey', 'chào')
greeting_responses= ['Chào bạn, rất vui được hỗ trợ cho bạn!',
                     'Xin chào, rất vui được hỗ trợ cho bạn!',
                     'Xin chào, tôi có thể cung cấp cho bạn một số thông tin về đại dịch Covid-19.',
                     'Chào bạn, tôi có thể cung cấp cho bạn một số thông tin về đại dịch Covid-19.',
                     'Xin chào, rất mong các thông tin hỗ trợ của tôi sẽ giúp ích cho bạn!',
                     'Chào bạn, rất mong các thông tin hỗ trợ của tôi sẽ giúp ích cho bạn!']

# Các mẫu nói tục
swearing_inputs= ('đụ', 'địt', 'đéo', 'cặc', 'lồn', 'mẹ mày', 'bố mày', 'thằng', 'cha mày', 'chát sex')
swearing_responses= ['Tôi xin phép không trả lời vì câu hỏi của bạn thiếu nghiêm túc.',
                     'Bạn có thể hỏi nghiêm túc hơn được không?']

# tạm biệt
bye_inputs= ('tạm biệt', 'xin cảm ơn', 'cảm ơn bạn','cảm ơn', 'bye', 'thank', 'thanks')
bye_responses= ['Tạm biệt!',
                'Rất vui được hỗ trợ bạn!',
                'Rất mong được gặp lại!']

# Phản hồi đối với câu hỏi
def response(question_input):
    # Chuẩn hóa ký tự trong câu hỏi nhập vào
    question_input= standardized_sentence(question_input)
    question_input= [question_input]   
    # Tính tfidf với câu hỏi nhập vào
    tfidf_input= tfidf_vectorizer.transform(question_input)
    # Tính toán độ giống nhau của câu nhập vào và câu hỏi mẫu
    vals = cosine_similarity(tfidf_input, tfidf)
    vals= vals.flatten()
    max_val= vals.max()
    index= vals.argmax()
    # Phản hồi
    if max_val <= 0.4:
        response= random.choice(sorry_response)
    else:
        response= sample_answer[index]
    return response

def robot_brain(user_response):
    user_response= user_response.lower()
    if (True in [greeting_input in user_response for greeting_input in greeting_inputs]) and (user_response.count(' ') <= 6) or (user_response == 'hi'):
        return random.choice(greeting_responses)
    elif (True in [swearing_input in user_response for swearing_input in swearing_inputs]):
        return random.choice(swearing_responses)
    elif (True in [bye_input in user_response for bye_input in bye_inputs]) and (user_response.count(' ') <= 6):
        return random.choice(bye_responses)
    else:
        return response(user_response).replace('\\n', '\n')

# -------------------------- PHẦN WEB ----------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(robot_brain(userText))
