# Модели прогноза уровня заполняемости парковок

Для анализа взяты исторические данные о заполняемости автомобильных парковок г. Бирмингем. Проводится предварительная обработка и устранение технических ошибок в исходных данных, визуализация зависимостей между параметрами, сравнение показателей точности (коэффициент детерминации, средняя абсолютная ошибка) нескольких регрессионных моделей.

Источник данных: https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham

Объем выборки: 35717 записей

Временной интервал: октябрь-декабрь 2016 г.

Параметры:
- SystemCodeNumber: ID автомобильной парковки
- Capacity: вместимость паркивки (мест)
- Occupancy: уровень заполняемости (количество машин на парковке)
- LastUpdated: дата и время
