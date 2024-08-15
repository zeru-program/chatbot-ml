# import library
import string
import pickle
import numpy as np
import os
from fuzzywuzzy import process
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from util import JSONParser

def clear():
    os.system('cls')

def setup():
    print(f"""\033[33m
          ╔═╗┌─┐┌─┐┬ ┬┌─┐┌┬┐       ╔╦╗┌─┐┌─┐┬ ┬┬┌┐┌┌─┐  ╦  ┌─┐┌─┐┬─┐┌┐┌┬┌┐┌┌─┐
          ╔═╝├┤ │  ├─┤├─┤ │   ───  ║║║├─┤│  ├─┤││││├┤   ║  ├┤ ├─┤├┬┘││││││││ ┬
          ╚═╝└─┘└─┘┴ ┴┴ ┴ ┴        ╩ ╩┴ ┴└─┘┴ ┴┴┘└┘└─┘  ╩═╝└─┘┴ ┴┴└─┘└┘┴┘└┘└─┘\033[36m
        Zechat sebuah chat di cli menggunakan python. silakan gunakan dengan bijak.\033[0m
          """)

def preprocess(chat):
    # konversi ke non kapital
    chat = chat.lower()
    # hilangkan tanda baca
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat


def fuzzy_match(user_input, patterns, threshold=80):
    match, score = process.extractOne(user_input, patterns)
    return match if score >= threshold else None

def bot_response(chat, pipeline, jp):
    chat = preprocess(chat)
    res = pipeline.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.2:
        return "Maaf saya tidak mengerti apa yang anda bicarakan :(", None
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        # Optional: Fuzzy matching to enhance understanding
        patterns = jp.df.query(f'intents == "{pred_tag}"')['text_input'].tolist()
        matched_pattern = fuzzy_match(chat, patterns)
        if matched_pattern:
            return jp.get_response(pred_tag), pred_tag
        else:
            return "Maaf, saya tidak bisa memahami pertanyaan Anda dengan baik.", pred_tag

clear()
setup()

# load data
path = "data/intents.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# praproses data
# case folding -> transform kapital ke non kapital, hilangkan tanda baca
df['text_input_prep'] = df.text_input.apply(preprocess)

# pemodelan
pipeline = make_pipeline(CountVectorizer(),
                        MultinomialNB())

# train
# print("[INFO] Training Data ...")
pipeline.fit(df.text_input_prep, df.intents)

# save model
with open("model_chatbot.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

# interaction with bot
# print("[INFO] Anda Sudah Terhubung dengan Bot Kami")
while True:
    chat = input("Anda >> ")
    res, tag = bot_response(chat, pipeline, jp)
    print(f"ZeBot >> {res}")
    if tag == 'bye':
        break

