Анализ и кластеризация предложений по продаже женских платьев с AliExpress и данных о продажах.

Используемые модели:
- KMeans
- Birch
- AgglomerativeClustering

Производится анализ корреляции между числовыми параметрами, анализ распределения числовых и категорийных параметров, определение оптимального набора параметров для моделей (Principal Component Analysis), подбор оптимального количества кластеров, визуализация и интерпретация полученных результатов.

Источник данных: https://archive.ics.uci.edu/ml/datasets/Dresses_Attribute_Sales

Исходные данные извлекаются из двух таблиц (файлы формата xlsx):

I. Таблица с характеристиками товаров:

   Количество строк таблице: 500
   Количество параметров: 14

   Параметры и их значения:
   - Dress_ID: код товара
   - Style: стиль (Bohemia, brief, casual, cute, fashion, flare, novelty, OL, party, sexy, vintage, work)
   - Price: ценовой диапазон (Low, Average, Medium, High, Very-High)
   - Rating: рейтинг (1-5)
   - Size: размер (S, M, L, XL, Free)
   - Season: сезон (Autumn, winter, Spring, Summer)
   - NeckLine: форма выреза (O-neck, backless, board-neck, Bowneck, halter, mandarin-collor, open, peterpan-collor, ruffled, scoop, slash-neck, square-collar, sweetheart, turndowncollar, V-neck)
   - SleeveLength: длина рукава (full, half, halfsleeves, butterfly, sleveless, short, threequarter, turndown, null)
   - Waistline: талия (dropped, empire, natural, princess, null)
   - Material: материал (wool, cotton, mix, etc.)
   - FabricType: ткань (shafoon, dobby, popline, satin, knitted, jersey, flannel, corduroy, etc.)
   - Decoration: отделка (applique, beading, bow, button, cascading, crystal, draped, embroridary, feathers, flowers, etc.)
   - Pattern Type: рисунок (solid, animal, dot, leapard, etc.)
   - Recommendation: рекомендации (0, 1)

II. Таблица с данными о продажах:

   Количество строк таблице: 500
   Количество параметров: 24

   Параметры и их значения:
   - Dress_ID: код товара
   - Остальные столбцы представляют собой даты в формате дд.мм.гггг и дд/мм/гггг
