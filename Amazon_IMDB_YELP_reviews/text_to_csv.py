# Преобразуем три текстовых файла с отзывами и их категориями
# в один общий csv-файл, добавляя данные об источнике информации.

import csv

# Считываем данные из текстового файла с отзывами с Amazon:
with open('amazon_cells_labelled.txt', 'r') as f:
    contents = f.readlines()
    # Преобразуем каждую строку в список и добавляем источник информации:
    for line in contents:
        line = line.replace('\n', '').split('\t')
        line.append('Amazon')
        # Записываем обработанную строку в csv-файл:
        with open('reviews_data.csv', 'a', newline='') as new_f:
            csv_writer = csv.writer(new_f, delimiter='\t')
            csv_writer.writerow(line)

# Повторяем для текстового файла с отзывами с IMDB:
with open('imdb_labelled.txt', 'r') as f:
    contents = f.readlines()
    for line in contents:
        line = line.replace('\n', '').split('\t')
        line.append('IMDB')
        with open('reviews_data.csv', 'a', newline='') as new_f:
            csv_writer = csv.writer(new_f, delimiter='\t')
            csv_writer.writerow(line)

# Повторяем для текстового файла с отзывами с Yellow Pages:
with open('yelp_labelled.txt', 'r') as f:
    contents = f.readlines()
    for line in contents:
        line = line.replace('\n', '').split('\t')
        line.append('YELP')
        with open('reviews_data.csv', 'a', newline='') as new_f:
            csv_writer = csv.writer(new_f, delimiter='\t')
            csv_writer.writerow(line)
