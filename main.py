# import lib
import string
from util import JSONParser

def preprocess(chat):
    # connvert ke non kapital
    chat = chat.lower()
    tb = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tb)
    return chat

# load data
path = "data/intents.json"
jp = JSONParser()
jp.parser(path)
df = jp.get_dataframe()

# prosees bang
