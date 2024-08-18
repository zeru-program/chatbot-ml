import requests
from bs4 import BeautifulSoup


def scrape_data(query):
    # Misalnya, mencari data dari Wikipedia
    search_url = f"https://en.wikipedia.org/wiki/{query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text[:500]  # Mengambil 500 karakter pertama

# Penggunaan
print(scrape_data("Python (programming language)"))