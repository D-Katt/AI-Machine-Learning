Примеры классификации на основе 'Wine recognition dataset'.
Классификация выполняется с применением моделей LogisticRegression, KNN и SVC.
Для анализа ошибок в классификации на примере модели с ограниченным набором параметров
используется 'confusion_matrix', производится расчет показателей 'recall_score' и 'ROC AUC'.

Количество строк в базе: 178
Количество параметров: 13 числовых параметров и класс

Перечень параметров:
	- Alcohol
	- Malic acid
	- Ash
	- Alcalinity of ash
	- Magnesium
	- Total phenols
	- Flavanoids
	- Nonflavanoid phenols
	- Proanthocyanins
	- Color intensity
	- Hue
	- OD280/OD315 of diluted wines
	- Proline

Кодировка (перечень классов):
    - class_0
    - class_1
    - class_2

Распределение выборки по классам: class_0 (59), class_1 (71), class_2 (48)