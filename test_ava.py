import requests

url = "http://51.21.218.121:5000/evaluate"
# This one does not exists inside the storage of paintings
#file_path = "data/images/test/175e9250451079.58d10d89ef851.jpg"
#file_path = "data/images/test/images.jpg"
#file_path = "data/images/test/007.jpg"
#file_path = "data/images/test/289.JPG"
file_path = "data/images/test/1333.JPG"

# This one exists inside the storage of paintings
#file_path = "data/images/menorinka/Jonas-Mackevičius_Capri-salos-motyvas-su-žydinčia-visterija_48.jpg"
#file_path = "data/images/menorinka/Solomonas-Teitelbaumas_Kelias-pro-žydintį-sodą_63.jpg"
#file_path = "data/images/menorinka/Vytenis-Lingys_Dylantis-mėnulis_36.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())