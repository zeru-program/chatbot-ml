# import library
import string
import pickle
import numpy as np
import os
import json
import random
from datetime import datetime
from fuzzywuzzy import process
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from util import JSONParser

def clear():
    os.system('cls')

def setup():
    # dd/mm/YY H:M:S
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"""\033[33m
                                    {dt_string}
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
        chatBad = ["Maaf saya masih tahap pengembangan, belum mengerti apa yang anda inginkan", "Maaf saya tidak mengerti.", "Saya tidak mengerti apa yang anda bicarakan :(", "Yahh saya tidak mengerti, coba bahas topik lain bangg", "Sorry bang saya tidak  mengerti apa yang anda maksud. bisa jelaskan lebih lanjut topik mu?", "Maaf data saya masih terbatas sehingga belum mengerti apa yang anda maksud, kirimkan topik yang ingin kamu bahas ke akun ig @zerr.ace yuk"]
        if matched_pattern:
            return jp.get_response(pred_tag), pred_tag
        else:
          return random.choice(chatBad), pred_tag

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

# interaktif dengan bot
while True:
    chat = input(" Anda \033[0m    \033[93m>>\033[0m ")
    if chat == "q":
        print(f"""

 ------------------------- Anda telah keluar -------------------------
              """)
        empety = []
        with open("data/chat-user.json", 'w') as file:
          json.dump(empety, file)
        exit()
    if chat == "d":
        with open('data/chat-user.json', 'r') as file:
         loaded_data = json.load(file)
         for x in loaded_data:
             print(x)
    if chat == "":
       print(f" \033[46m \033[30mZeChat \033[0m \033[93m>>\033[0m Mohon ketik pesan dengan benar !")
    # Nama file JSON
    filename = 'data/chat-user.json'
    # Membaca data yang ada jika file sudah ada
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                # Jika file kosong atau tidak valid, mulai dengan list kosong
                data = []
    else:
        # Jika file belum ada, mulai dengan list kosong
        data = []

    # Menambahkan data baru ke list data
    data.append(chat)
    # Menyimpan data ke file JSON
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    res, tag = bot_response(chat, pipeline, jp)
    print(f" \033[46m \033[30mZeChat \033[0m \033[93m>>\033[0m {res}")
    if tag == 'bye':
        break


