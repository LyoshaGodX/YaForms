import json
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

file_path = 'YForms data.json'

with open(file_path, 'r', encoding='utf-8') as file:
    json_data = file.read()

data = json.loads(json_data)

for row in data:
    del row[:2]

education_levels = {
    "Общее": 0,
    "Среднее": 1,
    "Среднее специальное": 2,
    "Бакалавр": 3,
    "Магистр и выше": 4
}

income_levels = {
    "Меньше 20 000 рублей": 10000,
    "20 000 - 30 000 рублей": 25000,
    "30 000 - 40 000 рублей": 35000,
    "40 000 - 50 000 рублей": 45000,
    "50 000 - 80 000 рублей": 65000,
    "80 000 рублей и выше": 80000
}

sleep_duration_levels = {
    "Менее 5 часов": 3,
    "5-6 часов": 5.5,
    "6-7 часов": 6.5,
    "7-8 часов": 7.5,
    "8 часов и более": 9
}

physical_activity_levels = {
    "0": 0,
    "1-2": 1.5,
    "2-3": 2.5,
    "4-6": 5,
    "10 и больше": 10
}

water_consumption_levels = {
    "Меньше 1": 0.5,
    "1-1.5": 1.25,
    "1.5-2": 1.75,
    "2 и больше": 2.5
}

for rows in data:
    for row in rows:
        if row[0] == "Ваш возраст":
            row[1] = int(row[1])
        elif row[0] == "Ваш уровень образования":
            row[1] = education_levels.get(row[1], -1)
        elif row[0] == "Ваш ежемесячный доход":
            row[1] = income_levels.get(row[1], -1)
        elif row[0] == "Средняя продолжительно вашего сна":
            row[1] = sleep_duration_levels.get(row[1], -1)
        elif row[0] == "Оцените уровень вашего стресса / Уровень стресса в предыдущий месяц":
            row[1] = int(row[1])
        elif row[0] == "Оцените уровень вашего стресса / Уровень стресса в этом месяце":
            row[1] = int(row[1])
        elif row[0] == "Ваша физическая активность в неделю":
            row[1] = physical_activity_levels.get(row[1], -1)
        elif row[0] == "Сколько воды в день вы потребляете":
            row[1] = water_consumption_levels.get(row[1], -1)

processed_data = []
for rows in data:
    row_dict = {}
    for row in rows:
        row_dict[row[0]] = row[1]
    processed_data.append(row_dict)

data_df = pd.DataFrame(processed_data)

# Рассчет матрицы корреляции
correlation_matrix = data_df.corr()

# Вывод матрицы корреляции
print("Матрица корреляции:")
print(correlation_matrix)

# Итерация по всем парам признаков
for i in range(len(data_df.columns)):
    for j in range(i + 1, len(data_df.columns)):
        feature1 = data_df.columns[i]
        feature2 = data_df.columns[j]

        # Рассчет коэффициента корреляции Пирсона
        correlation_coefficient, _ = pearsonr(data_df[feature1], data_df[feature2])

        # Рассчет t-статистики и p-значения с использованием t-теста
        t_statistic, p_value = ttest_ind(data_df[feature1], data_df[feature2])

        # Вывод результатов
        print(f"\n{feature1} vs {feature2}:")
        print(f"Коэффициент корреляции Пирсона: {correlation_coefficient}")
        print(f"t-статистика: {t_statistic}")
        print(f"p-значение: {p_value}")

# Новые названия признаков (вопросов)
new_feature_names = ["Возраст", "Образование", "Доход", "Сон", "Стресс пр. месяц", "Стресс этот месяц", "Спорт", "Выпитая вода"]

# Настройка размера графика
plt.figure(figsize=(12, 8))

# Подготовка пустой матрицы для p-value
p_value_matrix = np.zeros_like(correlation_matrix, dtype=float)

# Итерация по всем парам признаков
for i in range(len(data_df.columns)):
    for j in range(i + 1, len(data_df.columns)):
        feature1 = data_df.columns[i]
        feature2 = data_df.columns[j]

        # Рассчет p-значения с использованием t-теста
        _, p_value = ttest_ind(data_df[feature1], data_df[feature2])

        # Запись p-значения в матрицу
        p_value_matrix[i, j] = p_value
        p_value_matrix[j, i] = p_value  # Симметричная матрица

# Построение тепловой карты корреляций с p-value и новыми названиями
heatmap = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5,
                      cbar_kws={'label': 'Коэффициент корреляции Пирсона'},
                      xticklabels=new_feature_names, yticklabels=new_feature_names)

# Вывод p-value на график
for i in range(p_value_matrix.shape[0]):
    for j in range(p_value_matrix.shape[1]):
        plt.text(j + 0.53, i + 0.3, f'r={correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')
        plt.text(j + 0.52, i + 0.65, f'p={p_value_matrix[i, j]:.2f}', ha='center', va='center', color='black')

# Отображение графика
plt.show()

# Настройка размера графика
plt.figure(figsize=(15, 10))

# Итерация по всем вопросам
for i, feature in enumerate(data_df.columns):
    # Подготовка данных для текущего вопроса
    responses = data_df[feature].value_counts().sort_index()

    # Создание столбчатой диаграммы
    plt.subplot(4, 2, i + 1)
    responses.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Ответы на вопрос: {new_feature_names[i]}')
    plt.xlabel('Ответы')
    plt.ylabel('Количество')

# Размещение графиков
plt.tight_layout()
plt.show()

strongest_correlations = []

for i in range(len(data_df.columns)):
    for j in range(i + 1, len(data_df.columns)):
        feature1 = data_df.columns[i]
        feature2 = data_df.columns[j]

        # Рассчет p-значения с использованием t-теста
        _, p_value = ttest_ind(data_df[feature1], data_df[feature2])

        # Добавление корреляции и p-value в список
        strongest_correlations.append((feature1, feature2, correlation_matrix.iloc[i, j], p_value))

# Сортировка по убыванию степени корреляции и восхождению p-value
strongest_correlations.sort(key=lambda x: (abs(x[2]), x[3]))

# Выбор топ 6
top_6_correlations = strongest_correlations[:6]

# Создание графиков для топ 6 корреляций
plt.figure(figsize=(15, 14))

for idx, (feature1, feature2, correlation, p_value) in enumerate(top_6_correlations):
    plt.subplot(3, 2, idx + 1)
    plt.scatter(data_df[feature1], data_df[feature2], alpha=0.7, color='blue')
    plt.title(f'Корреляция: {correlation:.2f}, p-value: {p_value:.4f}')
    plt.xlabel(new_feature_names[data_df.columns.get_loc(feature1)])
    plt.ylabel(new_feature_names[data_df.columns.get_loc(feature2)])

# Размещение графиков
plt.tight_layout()
plt.show()