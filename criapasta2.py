import requests
import os

def main():
    
    # GET LIST NAMES
    url = 'https://dog.ceo/api/breeds/list/all'
    racas_json = requests.get(url).json()
    racas_json = racas_json['message']
    print(racas_json)
    
    # GET IMAGES BY NAME
    for key, value in racas_json.items():
        
        if (value):
            for name_value in value:
                # GET IMAGE
                url = 'https://dog.ceo/api/breed/' + key + '/' + name_value + '/images/random'
                img = requests.get(url).json()
                
                # DOWNLOAD IMAGE
                download(key, name_value, img)
        else:
            # GET IMAGE
            url = 'https://dog.ceo/api/breed/' + key + '/images/random'
            img = requests.get(url).json()
            
            # DOWNLOAD IMAGE
            download(key, '', img)
            
                     
def download(name, subname, img):
        url = img['message']
        if (img['status'] == 'success'):
            img = requests.get(url)
            
            if (subname != ''):
                subname = '-' + subname
            
            if not os.path.exists(name + subname):
                os.makedirs(name + subname)
                print('Diretório criado com sucesso.')
                with open(name + subname  + '/' + name + '.jpg', 'wb') as file:
                     file.write(img.content)
                     print('Imagem salva com sucesso.')


main()
