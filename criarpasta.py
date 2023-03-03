import os
import json
import requests

url = "https://dog.ceo/api/breeds/list/all"

response = requests.get(url)

content = response.json()

json_string = json.dumps(content)

lista = json.loads(json_string)

lista1 = lista['message']

lista2 = list(lista1)
# Lista com os nomes das pastas a serem criadas
folders = lista2

# Itera sobre a lista e cria cada pasta
for folder in folders:
    os.mkdir(folder)
