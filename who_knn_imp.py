import csv

# CSV dosyasını açma ve okuma
with open('Life Expectancy Data.csv', mode='r', encoding='utf-8') as dosya:
    csv_okuyucu = csv.reader(dosya)
    for satir in csv_okuyucu:
        print(satir)  # Her bir satır bir liste olarak gelir
