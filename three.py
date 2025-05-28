#pip install polars
import polars as pl

def upload_full_db(path: str = "results/full_BD_isAlive.xlsx") -> pl.DataFrame:
    # Загружаем данные сразу в dataframe
    # Параметр path — путь к файлу с базой пациентов
    return pl.read_excel(path)

def load_F(path: str = "data/Показатель_F.xlsx") -> pl.DataFrame:
    # Оставляем только столбцы CaseID и Результат для ферритина
    return (
        pl.read_excel(path, columns=["CaseID", "Показатель_F", "Результат"])
        .filter(pl.col("Показатель_F") == "Ферритин")       # Фильтруем по названию показателя
        .select(["CaseID", "Результат"])                    # Оставляем только нужные поля
        .rename({"Результат": "F"})                         # Переименовываем столбец s "F"
        .with_columns(pl.col("F").cast(pl.Float64))         # Приводим значения к числу с плавающей точкой
        .drop_nulls()                                       # Удаляем строки с пустыми значениями
    )


def load_D(path: str = "data/Показатель_D.xlsx") -> pl.DataFrame:
    df = pl.read_excel(path, columns=["CaseID", "Показатель", "Результат_D"])
    df = df.filter(pl.col("Показатель").str.contains("D-димер"))
    df = df.with_columns(
        # Извлекаем чисто числовую часть из текста (удаляем нецифровые символы)
        pl.col("Результат_D")
        .str.replace_all(r"[^\d\.]", "")
        .alias("D_str")
    )
    df = df.filter(pl.col("D_str").str.contains(r"^\d+(\.\d+)?$"))
    # Приводим к Float64 и оставляем только CaseID и D
    return (
        df
        .with_columns(pl.col("D_str").cast(pl.Float64).alias("D"))
        .select(["CaseID", "D"])
    )


def load_vaccine_group(path: str, vaccine_name: str) -> pl.DataFrame:
    # Считываем весь лист с данными пациентов по конкретной вакцине
    df = pl.read_excel(path).select([
        "CaseID",   # уникальный ID пациента
        "Gender",   # пол пациента
        "Age",      # возраст пациента
        "Ther"      # тяжесть течения болезни
    ])

    # Добавляем колонку Vac — статус вакцинации, и Severity — уровень тяжести
    return df.with_columns(
        # Если vaccine_name == "Unvaccinated", отмечаем как «не вакцинирован», иначе указываем саму вакцину
        pl.when(pl.lit(vaccine_name) == "Unvaccinated")
        .then(pl.lit("не вакцинирован"))
        .otherwise(pl.lit(vaccine_name))
        .alias("Vac"),

        # Классификация Severity: «high» — строго «тяжелое», «medium» — всё остальное
        pl.when(
            pl.col("Ther").str.contains("тяжелое", literal=True) &
            ~pl.col("Ther").str.contains("среднетяжелое", literal=True)
        )
        .then(pl.lit("high"))
        .otherwise(pl.lit("medium"))
        .alias("Severity")
    )


# Функции формирования пар
def create_pairs(
        base_df: pl.DataFrame,
        F_df: pl.DataFrame,
        D_df: pl.DataFrame,
        gender: str
) -> pl.DataFrame:

    # Фильтрация и объединение данных по полу и объединение с F и D
    merged = (
        base_df
        .filter(pl.col("Gender") == gender)
        .join(F_df, on="CaseID", how="inner")
        .join(D_df, on="CaseID", how="inner")
        .sort("Age")
    )

    # Если после объединения нет данных — возвращаем пустой DataFrame
    if merged.height == 0:
        return pl.DataFrame()

    # Создание перекрестных пар (это клон, подготовка перекрёстного объединения)
    other = merged.clone().rename({
        "CaseID": "CaseID_2",
        "Age": "Age_2",
        "Vac": "Vac_2",
        "Severity": "Severity_2",
        "F": "F_2",
        "D": "D_2"
    })

    # Перекрёстное соединение и фильтрация по условиям:
    # — уникальные пары (ID1 < ID2)
    # — разница в возрасте ≤ 3 лет
    # — расхождение F и D ≤ 10%
    # — одинаковая степень тяжести
    # — если оба не вакцинированы, или одинаковая вакцина
    return (
        merged
        .join(other, how="cross")
        .filter(
            (pl.col("CaseID") < pl.col("CaseID_2")) &
            (abs(pl.col("Age") - pl.col("Age_2")) <= 3) &
            ((pl.col("F") - pl.col("F_2")).abs() / pl.col("F") <= 0.10) &
            ((pl.col("D") - pl.col("D_2")).abs() / pl.col("D") <= 0.10) &
            (pl.col("Severity") == pl.col("Severity_2")) &
            (       # условие: оба не вакцинированы
                    (pl.col("Vac") == "не вакцинирован") &
                    (pl.col("Vac_2") == "не вакцинирован")
            ) | (
                    # или одинковый тип вакцины
                    pl.col("Vac") == pl.col("Vac_2")
            )
        )
    )


def save_pairs(df: pl.DataFrame, filename: str):
    if df.height == 0:
        print(f"Нет данных для сохранения в {filename}")
        return

    df.rename({
        "CaseID": "ID_1",
        "CaseID_2": "ID_2",
        "Age": "Age_1",
        "Age_2": "Age_2",
        "Vac": "Vaccine_Status",
        "Severity": "Severity_Level",
        "F": "F_1",
        "F_2": "F_2",
        "D": "D_1",
        "D_2": "D_2"
    }).write_csv(filename)
    print(f"Сохранено {df.height} пар в файл {filename}")

if __name__ == "__main__":
    # Загрузка файлов
    full_db = upload_full_db()
    F_data = load_F()
    D_data = load_D()

    # Конфигурация путей к файлам по вакцинам
    VACCINE_CONFIG = {
        "SputnikV": "results/Спутник V.xlsx",
        "SputnikLight": "results/Спутник Лайт.xlsx",
        "EpiVacCorona": "results/Эпиваккорона.xlsx",
        "KoviVac": "results/Ковивак.xlsx",
        "Unvaccinated": "results/full_BD_isAlive.xlsx"
    }

    for vaccine_name, vaccine_path in VACCINE_CONFIG.items():
        try:
            # Загружаем пациентов данной группы
            vaccine_df = load_vaccine_group(vaccine_path, vaccine_name)
            print(f"\nОбработка: {vaccine_name}")
            print(f"Всего пациентов: {vaccine_df.height}")

            for gender in ("м", "ж"):
                print(f"\nПол: {gender}")
                # Фильтрация по полу
                gender_filtered = vaccine_df.filter(pl.col("Gender") == gender)
                print(f"После фильтрации по полу: {gender_filtered.height}")

                # Объединение с F и D
                merged = (
                    gender_filtered
                    .join(F_data, on="CaseID", how="inner")
                    .join(D_data, on="CaseID", how="inner")
                )
                print(f"После объединения с F/D: {merged.height}")

                # Создание пар
                pairs = create_pairs(
                    base_df=vaccine_df,
                    F_df=F_data,
                    D_df=D_data,
                    gender=gender
                )

                save_pairs(
                    pairs,
                    f"results/pairs_{vaccine_name}_{gender}.csv"
                )

        except Exception as e:
            print(f"Ошибка при обработке {vaccine_name}: {str(e)}")