Модель прогнозирует число въездных туристских поездок иностранных граждан в Российскую Федерацию на 3 года вперед на основе данных за предшествующие периоды.
Используется модель ARIMA модуля statsmodels.

Источник данных: https://www.fedstat.ru/indicator/59466

В файле 'foreign_tourists.csv' содержится предварительно обработанная база данных.

Параметры базы данных:
 - Year: числовое значение, год
 - Period: текстовое значение, определяющее период в течение года (январь-март, январь-июнь, январь-сентябрь, январь-декабрь)
 - Date: объект, производное значение от параметра 'Period', соответствующее последнему календарному числу соответствующего периода
 - Foreign_tourists: числовое значение, количество иностранных туристов