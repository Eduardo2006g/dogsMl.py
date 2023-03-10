import requests
import os


def main():
    # quantidade de imagens por pasta
    qtd = 10

    # pega os nomes da API de lista
    url = 'https://dog.ceo/api/breeds/list/all'
    racas_json = requests.get(url).json()
    racas_json = racas_json['message']

    # separa as imagens por nome
    for key, value in racas_json.items():

        if (value):
            for name_value in value:
                url = 'https://dog.ceo/api/breed/' + key + '/' + \
                    name_value + '/images/random/' + str(qtd)
                img = requests.get(url).json()

                # baixa a imagem
                download(key, name_value, img)
        else:
            # pega imagem de raça
            url = 'https://dog.ceo/api/breed/' + \
                key + '/images/random/' + str(qtd)
            img = requests.get(url).json()

            # baixa imgagem
            download(key, '', img)


# download das raças e sub raças

def download(name, subname, req):
    url = req['message']
    if (req['status'] == 'success'):

        # verifica se é raca ou subraca
        if (subname != ''):
            subname = '-' + subname

        # cria pasta se não existir
        if (not os.path.exists(name + subname)):
            os.makedirs(name + subname)

        for link in url:
            img = requests.get(link)

            # nome da imagem.jpg
            name_file = 'https://images.dog.ceo/breeds/' + name + subname + '/'
            name_file = link[len(name_file)::]

            # salva imagem
            with open(name + subname + '/' + name_file, 'wb') as file:
                file.write(img.content)


main()
