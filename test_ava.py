import requests

url = "http://51.20.5.62:5000/evaluate"
# This one does not exists inside the storage of paintings
#file_path = "data/images/test/175e9250451079.58d10d89ef851.jpg"
#file_path = "data/images/test/images.jpg"
#file_path = "data/images/test/007.jpg"
#file_path = "data/images/test/289.JPG"
#file_path = "data/images/test/1333.JPG"
file_path = "data/images/artprice/agata_bogacka-rzeczywlscle,_mkodzl_sa,_realistami_(przed_lustrem)-86.jpg"

# This one exists inside the storage of paintings
#file_path = "data/images/menorinka/Jonas-Mackevičius_Capri-salos-motyvas-su-žydinčia-visterija_48.jpg"
#file_path = "data/images/menorinka/Solomonas-Teitelbaumas_Kelias-pro-žydintį-sodą_63.jpg"
#file_path = "data/images/menorinka/Vytenis-Lingys_Dylantis-mėnulis_36.jpg"
#file_path = "data/images/menorinka/Albert-Graefle_Moters-portretas_81.jpg"
#file_path = "data/images/menorinka/Algis-Skačkauskas_Fleitistės-ir-nukryžiuotas_48.jpg"
#file_path = "data/images/menorinka/Algimantas-Kuras_Kavinės-lankytoja_80.jpg"
#file_path = "data/images/menorinka/Vincas-Norkus_Katedra-ir-Varpinė-nuo-Šventaragio-gatvės-pusės_32.jpg"
#file_path = "data/images/menorinka/Adomas-Varnas-(atrib.)_Jono-Basanavičiaus-portretas_34.jpg"
#file_path = "data/images/menorinka/Algimantas-Jonas-Kuras_Mergaitės-portretas-arkų-fone_12.jpg"
#file_path = "data/images/menorinka/Adomas-Galdikas_Abstrakcija_53.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())