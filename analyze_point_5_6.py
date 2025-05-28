import pandas as pd
import numpy as np
import matplotlib # Добавляем этот импорт
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, spearmanr
#chi2_contingency - Функция для выполнения теста хи-квадрат на независимость признаков в таблице сопряженности
# mannwhitneyu - U-критерий Манна-Уитни. Это непараметрический тест для сравнения двух независимых выборок
# kruskal - Критерий Краскела-Уоллиса. Это непараметрический тест для сравнения трех и более независимых выборок
# spearmanr - Функция для вычисления коэффициента корреляции Спирмена между двумя переменными
import os 
import glob 


def preprocess_data(df):
    """Допустимая обработка для DataFrame с парами пациентов"""
    # Преобразование колонок Start и End в datetime
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['End'] = pd.to_datetime(df['End'], errors='coerce')
    
    # Расчет длительности госпитализации
    if 'Start' in df.columns and 'End' in df.columns and df['Start'].notna().all() and df['End'].notna().all():
        df['Hospitalization_Duration'] = (df['End'] - df['Start']).dt.days
    else:
        print("Warning: Не удалось расчитать 'Hospitalization_Duration' из-за отсутствубщих или неверных дат 'Start'/'End'")
        df['Hospitalization_Duration'] = np.nan

    #Создание бинарной переменной исхода (к примеру, 1 для 'Умер', 0 для 'Выписан')
    df['Outcome_Binary'] = df['Outcome'].map({'Умер': 1, 'Выписан': 0}).astype('Int64')
    
    return df

def analyze_point_5(paired_df):
    """
    Пункт 5: Сравнить полученные группы по исходам, визуалировать
    - жив/умер
    - для живых - быстрее ли выздоравливали (=сроки госпитализации)
    """
    print("====== Анализ: Сравнение групп по исходам ======")
    
    if 'Therapy_Group' not in paired_df.columns or 'Outcome' not in paired_df.columns:
        print("Error: Отсутствует колонка 'Therapy_Group' или 'Outcome' в paired_df.")
        return
    
    if paired_df['Therapy_Group'].nunique() < 2:
        print("Error: Для сравнения необходимо как минимум две терапевтические группы в данных.")
        return

    # 5a: Outcome ( жив/умер )
    print("\n5a: Outcome (жив/умер) по группам терапии")
    outcome_counts = pd.crosstab(paired_df['Therapy_Group'], paired_df['Outcome'])
    print("Количества:\n", outcome_counts)
    
    if outcome_counts.empty:
        print("Нет данных для анализа исходов по группам терапии.")
        return

    outcome_percentages = pd.crosstab(paired_df['Therapy_Group'], paired_df['Outcome'], normalize='index') * 100
    print("\nПроценты:\n", outcome_percentages)
    
    # Визуализация для 5a
    outcome_percentages.plot(kind='bar', figsize=(10, 6))
    plt.title('Процентное соотношение исходов по группам терапии')
    plt.ylabel('Процент пациентов (%)')
    plt.xlabel('Группа терапии')
    plt.xticks(rotation=0)
    plt.legend(title='Исход')
    plt.tight_layout()
    plt.show()
    
    # Статистический тест
    if outcome_counts.shape[0] > 1 and outcome_counts.shape[1] > 1:
        try:
            chi2, p, dof, expected = chi2_contingency(outcome_counts)
            print(f"\nТест хи-квадрат для исхода по группам терапии: хи-квадрат={chi2:.2f}, p-value={p:.3f}")
            if p < 0.05:
                print("Разница в исходах между группами терапии статистически значима.")
            else:
                print("Статистически значимой разницы в исходах между группами терапии не обнаружено.")
        except ValueError as e:
            print(f"\nНе удалось выполнить тест хи-квадрат: {e}")
            print("Это может произойти, если ожидаемые частоты слишком малы.")
    else:
        print("\nНедостаточно данных (различных групп или исходов) для выполнения теста хи-квадрат для исходов по группам терапии.")

    # 5b: Длительность госпитализации для выживших
    print("\n5b: Длительность госпитализации для выживших по группам терапии")
    survivors_df = paired_df[paired_df['Outcome'] == 'Выписан'].copy()
    
    if 'Hospitalization_Duration' not in survivors_df.columns or survivors_df['Hospitalization_Duration'].isna().all():
        print("Warning: 'Hospitalization_Duration' недоступна или все значения NaN для выживших. Пропуск этой части.")
    elif survivors_df.empty:
        print("Выжившие пациенты не найдены для анализа длительности госпитализации.")
    elif survivors_df['Therapy_Group'].nunique() < 2:
        print("Недостаточно терапевтических групп среди выживших для сравнения длительности госпитализации.")
    else:
        duration_summary = survivors_df.groupby('Therapy_Group')['Hospitalization_Duration'].agg(['mean', 'median', 'std', 'count'])
        print("Сводка по длительности госпитализации для выживших:")
        print(duration_summary)
        
        # Визуализация для 5b
        if not survivors_df['Hospitalization_Duration'].isna().all() and survivors_df['Therapy_Group'].nunique() > 0 :
            plt.figure(figsize=(8, 6))
            
            # Используем stripplot для отображения каждой точки
            # Увеличим размер точек для лучшей видимости
            sns.stripplot(x='Therapy_Group', y='Hospitalization_Duration', hue='Therapy_Group', 
                          data=survivors_df, jitter=0.1, size=8, alpha=0.7, legend=False, palette="Set2")

            # добавляем горизонтальные линии для медиан
            medians = survivors_df.groupby('Therapy_Group')['Hospitalization_Duration'].median()
            for i, group_name in enumerate(survivors_df['Therapy_Group'].unique()):
                if group_name in medians: # Проверяем, есть ли медиана для этой группы
                    plt.hlines(medians[group_name], xmin=i-0.4, xmax=i+0.4, 
                               color='gray', linestyle='--', linewidth=2,
                               label=f'Медиана {group_name}' if i == 0 else "_nolegend_") # Легенда только для первой

            plt.title('Длительность госпитализации для выживших (каждая точка - пациент)', fontsize=14)
            plt.ylabel('Длительность (дни)', fontsize=12)
            plt.xlabel('Группа терапии', fontsize=12)
            
            # Добавляем количество наблюдений (n) к меткам на оси X
            ax = plt.gca()
            current_labels = [label.get_text() for label in ax.get_xticklabels()]
            group_counts = survivors_df['Therapy_Group'].value_counts().reindex(current_labels)
            new_xticklabels = [f"{label} (n={group_counts.get(label, 0)})" for label in current_labels]
            ax.set_xticklabels(new_xticklabels)
            
            if len(survivors_df['Therapy_Group'].unique()) > 1 and any(group_name in medians for group_name in survivors_df['Therapy_Group'].unique()):
                 plt.legend(title="Медианы", loc='upper right')

            plt.tight_layout()
            plt.show()
        
        
        # Статистический тест для длительности госпитализации
        therapy_groups = survivors_df['Therapy_Group'].dropna().unique()
        if len(therapy_groups) == 2:
            group1_duration = survivors_df[survivors_df['Therapy_Group'] == therapy_groups[0]]['Hospitalization_Duration'].dropna()
            group2_duration = survivors_df[survivors_df['Therapy_Group'] == therapy_groups[1]]['Hospitalization_Duration'].dropna()
            
            if not group1_duration.empty and not group2_duration.empty:
                try:
                    stat, p_mw = mannwhitneyu(group1_duration, group2_duration, alternative='two-sided')
                    print(f"\nU-критерий Манна-Уитни для длительности госпитализации ({therapy_groups[0]} vs {therapy_groups[1]}): U-статистика={stat:.2f}, p-value={p_mw:.3f}")
                    if p_mw < 0.05:
                        print("Разница в длительности госпитализации между группами статистически значима.")
                    else:
                        print("Статистически значимой разницы в длительности госпитализации не обнаружено.")
                except ValueError as e:
                     print(f"\nНе удалось выполнить U-критерий Манна-Уитни: {e}")
            else:
                print("\nНедостаточно данных в одной или обеих группах для U-критерия Манна-Уитни по длительности госпитализации.")
        elif len(therapy_groups) > 2:
            print("\nОбнаружено более двух групп терапии; рассмотрите критерий Краскела-Уоллиса для сравнения длительности.")
        else: # Handles len(therapy_groups) < 2
            print("\nНедостаточно групп терапии с данными о выживших для сравнения длительности госпитализации.")

def print_section_header(title):
    """Вспомогательная функция для печати заголовков секций."""
    print(f"\n{'='*15} {title} {'='*15}")

def print_statistical_test_result(test_name, p_value, stat_value=None, threshold=0.05, extra_info=""):
    """Вспомогательная функция для печати результатов стат. тестов."""
    if stat_value is not None:
        print(f"{test_name}: Статистика={stat_value:.3f}, p-value={p_value:.4f}")
    else:
        print(f"{test_name}: p-value={p_value:.4f}")
    
    if p_value < threshold:
        print(f"Результат статистически значим (p < {threshold}). {extra_info}")
    else:
        print(f"Результат статистически не значим (p >= {threshold}). {extra_info}")

# ====== Функции для анализа влияния на исход заболевания (Outcome_Binary) ======

def analyze_numeric_vs_outcome(df, numeric_col, col_name_rus):
    print_section_header(f"6.1.x {col_name_rus} vs Исход (0-Выписан, 1-Умер)")
    if numeric_col not in df.columns or 'Outcome_Binary' not in df.columns:
        print(f"Отсутствуют необходимые столбцы: {numeric_col} или Outcome_Binary.")
        return

    df_clean = df.dropna(subset=[numeric_col, 'Outcome_Binary'])
    if df_clean.empty:
        print(f"Нет данных после удаления пропусков для {numeric_col} и Outcome_Binary.")
        return

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Outcome_Binary', y=numeric_col, data=df_clean, palette="coolwarm", inner="quartile") # ИЗМЕНЕНО ЗДЕСЬ
    plt.title(f'Распределение "{col_name_rus}" по исходам')
    plt.xlabel('Исход')
    plt.ylabel(col_name_rus)
    plt.xticks([0, 1], ['Выписан (0)', 'Умер (1)'])
    plt.tight_layout()
    plt.show()
    plt.close()

    group_survived = df_clean[df_clean['Outcome_Binary'] == 0][numeric_col]
    group_died = df_clean[df_clean['Outcome_Binary'] == 1][numeric_col]

    if not group_survived.empty:
        print(f"Статистика для '{col_name_rus}' (Выписаны, n={len(group_survived)}): "
              f"Среднее={group_survived.mean():.2f}, Медиана={group_survived.median():.2f}, "
              f"Мин={group_survived.min():.2f}, Макс={group_survived.max():.2f}")
    if not group_died.empty:
        print(f"Статистика для '{col_name_rus}' (Умерли, n={len(group_died)}): "
              f"Среднее={group_died.mean():.2f}, Медиана={group_died.median():.2f}, "
              f"Мин={group_died.min():.2f}, Макс={group_died.max():.2f}")

    if len(group_survived) > 1 and len(group_died) > 1:
        try:
            stat, p_val = mannwhitneyu(group_survived, group_died, alternative='two-sided')
            print_statistical_test_result(f"U-критерий Манна-Уитни для '{col_name_rus}'", p_val, stat)
        except ValueError as e:
            print(f"Ошибка при расчете U-критерия для '{col_name_rus}': {e}")
    else:
        print(f"Недостаточно данных в одной или обеих группах исходов для сравнения '{col_name_rus}'.")

def analyze_categorical_vs_outcome(df, cat_col, col_name_rus):
    print_section_header(f"6.1.x {col_name_rus} vs Исход (0-Выписан, 1-Умер)")
    if cat_col not in df.columns or 'Outcome_Binary' not in df.columns:
        print(f"Отсутствуют необходимые столбцы: {cat_col} или Outcome_Binary.")
        return

    df_clean = df.dropna(subset=[cat_col, 'Outcome_Binary'])
    if df_clean.empty:
        print(f"Нет данных после удаления пропусков для {cat_col} и Outcome_Binary.")
        return
    
    contingency_table = pd.crosstab(df_clean[cat_col], df_clean['Outcome_Binary'])
    print(f"\nТаблица сопряженности для '{col_name_rus}' и Исхода (0-Выписан, 1-Умер):\n{contingency_table}")

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        print("Недостаточно категорий или исходов для теста хи-квадрат.")
        return

    # Визуализация: процент умерших в каждой категории
    outcome_proportions = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
    if 1 in outcome_proportions.columns: # Если есть умершие
        outcome_proportions_to_plot = outcome_proportions[[1]].rename(columns={1: '% Умерших'})
        outcome_proportions_to_plot.plot(kind='bar', figsize=(10, 7), colormap="coolwarm_r")
        plt.title(f'Процент умерших по категориям "{col_name_rus}"')
        plt.ylabel('Процент умерших (%)')
        plt.xlabel(col_name_rus)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        plt.close()
    else: # Если все выжили
        print("Все пациенты в данной подгрупке выжили, график процента умерших не строится.")


    try:
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        print_statistical_test_result(f"Тест хи-квадрат для '{col_name_rus}'", p_val, chi2)
        if (expected < 5).any().any():
            print("Предупреждение: Некоторые ожидаемые частоты в тесте хи-квадрат < 5. Результаты могут быть неточными.")
    except ValueError as e:
        print(f"Ошибка при расчете теста хи-квадрат для '{col_name_rus}': {e}")

# ====== Функции для анализа влияния на длительность госпитализации (для выживших) ======

def analyze_numeric_vs_duration(df_survivors, numeric_col, col_name_rus):
    print_section_header(f"6.2.x {col_name_rus} vs Длительность госпитализации")
    if numeric_col not in df_survivors.columns or 'Hospitalization_Duration' not in df_survivors.columns:
        print(f"Отсутствуют необходимые столбцы: {numeric_col} или Hospitalization_Duration.")
        return

    df_clean = df_survivors.dropna(subset=[numeric_col, 'Hospitalization_Duration'])
    if df_clean.empty or len(df_clean) < 2: # Нужно хотя бы 2 точки для корреляции
        print(f"Нет данных или недостаточно данных после удаления пропусков для {numeric_col} и Hospitalization_Duration.")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=numeric_col, y='Hospitalization_Duration', data=df_clean, alpha=0.5)
    plt.title(f'"{col_name_rus}" vs Длительность госпитализации')
    plt.xlabel(col_name_rus)
    plt.ylabel('Длительность госпитализации (дни)')
    plt.tight_layout()
    plt.show()
    plt.close()

    try:
        corr, p_val = spearmanr(df_clean[numeric_col], df_clean['Hospitalization_Duration'])
        print_statistical_test_result(f"Корреляция Спирмена для '{col_name_rus}' и Длительности", p_val, corr, 
                                      extra_info=f"Коэффициент корреляции: {corr:.3f}")
    except ValueError as e:
        print(f"Ошибка при расчете корреляции для '{col_name_rus}': {e}")


def analyze_categorical_vs_duration(df_survivors, cat_col, col_name_rus):
    print_section_header(f"6.2.x {col_name_rus} vs Длительность госпитализации")
    if cat_col not in df_survivors.columns or 'Hospitalization_Duration' not in df_survivors.columns:
        print(f"Отсутствуют необходимые столбцы: {cat_col} или Hospitalization_Duration.")
        return

    df_clean = df_survivors.dropna(subset=[cat_col, 'Hospitalization_Duration'])
    if df_clean.empty:
        print(f"Нет данных после удаления пропусков для {cat_col} и Hospitalization_Duration.")
        return
    
    # Проверка на количество уникальных значений в категориях для Kruskal-Wallis
    category_counts = df_clean[cat_col].value_counts()
    valid_categories = category_counts[category_counts > 1].index # Категории с более чем 1 наблюдением
    
    if len(valid_categories) < 2:
        print(f"Недостаточно категорий с >1 наблюдением в '{col_name_rus}' для сравнения длительности.")
        if not df_clean.empty:
             plt.figure(figsize=(10, 7))
             sns.violinplot(x=cat_col, y='Hospitalization_Duration', data=df_clean, palette="viridis", inner="quartile")
             plt.title(f'Длительность госпитализации по категориям "{col_name_rus}"')
             plt.xlabel(col_name_rus)
             plt.ylabel('Длительность госпитализации (дни)')
             plt.xticks(rotation=45, ha="right")
             plt.tight_layout()
             plt.show()
             plt.close()
        return

    df_filtered_for_test = df_clean[df_clean[cat_col].isin(valid_categories)]
    
    plt.figure(figsize=(10, 7))
    sns.violinplot(x=cat_col, y='Hospitalization_Duration', data=df_filtered_for_test, palette="viridis", order=valid_categories.sort_values(), inner="quartile")
    plt.title(f'Длительность госпитализации по категориям "{col_name_rus}"')
    plt.xlabel(col_name_rus)
    plt.ylabel('Длительность госпитализации (дни)')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    plt.close()

    groups = [group['Hospitalization_Duration'].values for name, group in df_filtered_for_test.groupby(cat_col)]
    
    if len(groups) >= 2: # Kruskal-Wallis для >= 2 групп, Mann-Whitney U для 2 групп
        if len(groups) == 2:
            try:
                stat, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                print_statistical_test_result(f"U-критерий Манна-Уитни для '{col_name_rus}'", p_val, stat)
            except ValueError as e:
                print(f"Ошибка при расчете U-критерия для '{col_name_rus}': {e}")
        else: # len(groups) > 2
            try:
                stat, p_val = kruskal(*groups)
                print_statistical_test_result(f"Тест Краскела-Уоллиса для '{col_name_rus}'", p_val, stat)
            except ValueError as e:
                print(f"Ошибка при расчете теста Краскела-Уоллиса для '{col_name_rus}': {e}")
    else:
        print(f"Недостаточно групп для сравнения '{col_name_rus}'.")


def analyze_point_6(df_full_processed):
    print_section_header("Пункт 6: Влияние факторов на исходы")
    
    if df_full_processed.empty:
        print("Нет данных для анализа пункта 6.")
        return

    df_analysis = df_full_processed.copy()

    # ====== 6.1 Влияние факторов на исход (Умер/Выписан) ======
    print_section_header("6.1 Влияние факторов на ИСХОД (0-Выписан, 1-Умер)")

    analyze_numeric_vs_outcome(df_analysis, 'Age', 'Возраст')
    analyze_categorical_vs_outcome(df_analysis, 'Gender', 'Пол')
    analyze_categorical_vs_outcome(df_analysis, 'Vac', 'Статус вакцинации')
    analyze_categorical_vs_outcome(df_analysis, 'Severity_Group', 'Степень тяжести')
    analyze_numeric_vs_outcome(df_analysis, 'D_Value', 'D-димер')
    analyze_numeric_vs_outcome(df_analysis, 'F_Value', 'Ферритин')

    # ====== 6.2 Влияние факторов на длительность госпитализации (для выживших) ======
    print_section_header("6.2 Влияние факторов на ДЛИТЕЛЬНОСТЬ ГОСПИТАЛИЗАЦИИ (для выживших)")
    survivors_df = df_analysis[df_analysis['Outcome_Binary'] == 0].copy()

    if survivors_df.empty:
        print("Нет выживших пациентов для анализа длительности госпитализации.")
    else:
        print(f"Анализ длительности госпитализации для {len(survivors_df)} выживших пациентов.")
        analyze_numeric_vs_duration(survivors_df, 'Age', 'Возраст')
        analyze_categorical_vs_duration(survivors_df, 'Gender', 'Пол')
        analyze_categorical_vs_duration(survivors_df, 'Vac', 'Статус вакцинации')
        analyze_categorical_vs_duration(survivors_df, 'Severity_Group', 'Степень тяжести')
        analyze_numeric_vs_duration(survivors_df, 'D_Value', 'D-димер')
        analyze_numeric_vs_duration(survivors_df, 'F_Value', 'Ферритин')
    
    print_section_header("Анализ пункта 6 завершен")


def get_therapy_group_from_string(ther_string):
    """Определяет группу терапии на основе описания."""
    if pd.isna(ther_string):
        return None
    if "с ЛП" in ther_string:  # Пациенты, получавшие лекарственный препарат
        return "Intervention"
    elif "без ЛП" in ther_string:  # Пациенты, не получавшие лекарственный препарат
        return "Control"
    return None # Если не удается определить

def main():
    pairs_folder = "results/"
    master_file_path = os.path.join(pairs_folder, "full_BD_isAlive.xlsx")

    all_patient_records = []
    current_df_to_analyze = pd.DataFrame() # Инициализируем пустым DataFrame

    try:
        master_df = pd.read_excel(master_file_path, usecols=['CaseID', 'Start', 'End', 'Outcome'])
        if master_df.empty:
            print(f"Ошибка: Файл {master_file_path} пуст.")
            return 
        if 'CaseID' not in master_df.columns:
            print(f"Ошибка: В файле {master_file_path} отсутствует колонка 'CaseID'.")
            return 
        master_df.drop_duplicates(subset=['CaseID'], inplace=True) 
    except FileNotFoundError:
        print(f"Ошибка: Файл {master_file_path} не найден.")
        return 
    except Exception as e:
        print(f"Ошибка при чтении файла {master_file_path}: {e}")
        return

    pair_files_pattern = os.path.join(pairs_folder, "pairs_*.csv")
    all_pairs_files = glob.glob(pair_files_pattern)

    if not all_pairs_files:
        print(f"Файлы пар не найдены по шаблону: {pair_files_pattern}")
    else:
        print(f"Найдены следующие файлы с парами: {all_pairs_files}")
        for f_path in all_pairs_files:
            try:
                pair_df = pd.read_csv(f_path)
                print(f"Обработка файла: {f_path}, найдено строк: {len(pair_df)}")
                for _, row in pair_df.iterrows():
                    p1_therapy_group = get_therapy_group_from_string(row.get('Ther'))
                    p2_therapy_group = get_therapy_group_from_string(row.get('Ther_right'))

                    if (p1_therapy_group == "Intervention" and p2_therapy_group == "Control"):
                        all_patient_records.append({
                            'CaseID': row['ID_1'], 'Gender': row['Gender'], 'Age': row['Age_1'],
                            'Therapy_Group': "Intervention", 'Vac': row['Vaccine_Status'],
                            'D_Value': row['D_1'], 'F_Value': row['F_1'],
                            'Severity_Group': row['Severity_Level']
                        })
                        all_patient_records.append({
                            'CaseID': row['ID_2'], 'Gender': row['Gender_right'], 'Age': row['Age_2'],
                            'Therapy_Group': "Control", 'Vac': row['Vac_2'],
                            'D_Value': row['D_2'], 'F_Value': row['F_2'],
                            'Severity_Group': row['Severity_2']
                        })
                    elif (p1_therapy_group == "Control" and p2_therapy_group == "Intervention"):
                        all_patient_records.append({
                            'CaseID': row['ID_1'], 'Gender': row['Gender'], 'Age': row['Age_1'],
                            'Therapy_Group': "Control", 'Vac': row['Vaccine_Status'],
                            'D_Value': row['D_1'], 'F_Value': row['F_1'],
                            'Severity_Group': row['Severity_Level']
                        })
                        all_patient_records.append({
                            'CaseID': row['ID_2'], 'Gender': row['Gender_right'], 'Age': row['Age_2'],
                            'Therapy_Group': "Intervention", 'Vac': row['Vac_2'],
                            'D_Value': row['D_2'], 'F_Value': row['F_2'],
                            'Severity_Group': row['Severity_2']
                        })
            except Exception as e:
                print(f"Ошибка при обработке файла {f_path}: {e}")
        
        if not all_patient_records:
            print("Не найдено подходящих записей пациентов из файлов пар.")
        else:
            df_from_pairs = pd.DataFrame(all_patient_records)
            df_from_pairs.drop_duplicates(subset=['CaseID', 'Therapy_Group'], inplace=True)

            current_df_to_analyze = pd.merge(df_from_pairs, master_df, on='CaseID', how='left')
            
            missing_master_data = current_df_to_analyze[current_df_to_analyze['Outcome'].isna()]['CaseID'].unique()
            if len(missing_master_data) > 0:
                print(f"Предупреждение: Для {len(missing_master_data)} CaseID не найдена информация (Outcome/Start/End) в {master_file_path}.")
                current_df_to_analyze.dropna(subset=['Outcome', 'Start', 'End'], inplace=True)
    
    if not current_df_to_analyze.empty:
        print(f"\nВсего записей для анализа: {len(current_df_to_analyze)}")
        print("Распределение по Therapy_Group:\n", current_df_to_analyze['Therapy_Group'].value_counts())

        paired_df_processed = preprocess_data(current_df_to_analyze.copy())
        analyze_point_5(paired_df_processed)
        analyze_point_6(paired_df_processed)
        
        print("\n--- Скрипт анализа завершен ---")
    else:
        print("Нет данных для анализа. Пожалуйста, проверьте файлы пар и full_BD_isAlive.xlsx.")

if __name__ == "__main__":
    main()