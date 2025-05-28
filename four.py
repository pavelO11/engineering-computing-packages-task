import os
import polars as pl
from glob import glob
from scipy import stats # импортируем t-test и Wilcoxon из scipy.stats

# Пороги
AGE_DELTA = 3                  # максимальная разница в возрасте для нормализации
REL_DIFF = 0.10                # относительная погрешность для F и D
SIZE_THRESHOLD = 16 * 1024**3  # 16 ГБ — если файл больше, читаем только первые N строк
MAX_ROWS = 26_000_000          # максимум строк для больших файлов

def load_and_clean(path: str) -> pl.DataFrame:
    file_size = os.path.getsize(path)
    n_rows = None if file_size <= SIZE_THRESHOLD else MAX_ROWS  # отсекаем объёмные файлы

    df = pl.read_csv(path, n_rows=n_rows)
    return (
        df
        .filter(pl.col("ID_1") != pl.col("ID_2"))                # убираем самосравнения
        .with_columns([
            pl.min_horizontal(["ID_1","ID_2"]).alias("_i1"),     # сортировка пар для дедупликации
            pl.max_horizontal(["ID_1","ID_2"]).alias("_i2"),
        ])
        .unique(subset=["_i1","_i2"])                             # убираем дубликаты пар
        .drop(["_i1","_i2"])                                      # убираем временные столбцы
    )


def build_matched_pairs(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        ((pl.col("Age_1") - pl.col("Age_2")).abs() / AGE_DELTA).alias("s_age"),    # нормализуем по возрасту
        ((pl.col("F_1") - pl.col("F_2")).abs() / pl.col("F_1")).alias("s_f"),      # относительная разница F
        ((pl.col("D_1") - pl.col("D_2")).abs() / pl.col("D_1")).alias("s_d"),      # относительная разница D
    ]).with_columns([
        (pl.col("s_age") + pl.col("s_f") + pl.col("s_d")).alias("score")          # сводный скор — чем меньше, тем ближе
    ]).sort("score")                                                               # сортируем по лучшей схожести

    selected, used = [], set()
    for row in df.iter_rows(named=True):
        a, b = row["ID_1"], row["ID_2"]
        if a not in used and b not in used:         # жадный отбор — не используем повторно ID
            selected.append(row)
            used.update((a,b))

    return pl.DataFrame(selected).select([
        c for c in df.columns if c not in ("s_age","s_f","s_d","score")           # возвращаем только оригинальные колонки
    ])


def evaluate_group(df: pl.DataFrame, alpha: float = 0.05) -> dict:
    # Кол-во пар в текущей группе
    n = df.height

    # Если пар меньше двух — нормальной статистики не получится
    if n < 2:
        # Возвращаем метки: p-значения = None, сравнения считаем незначимыми
        return {
            "Pairs": n,
            **{k: (None if k.startswith("p_") else True)
               for k in ("p_age", "Age_ns", "p_f", "F_ns", "p_d", "D_ns")}
        }

    # Преобразуем в numpy-массивы — нужно для SciPy
    import numpy as np
    age1, age2 = df["Age_1"].to_numpy(), df["Age_2"].to_numpy()
    f1,   f2   = df["F_1"].to_numpy(),   df["F_2"].to_numpy()
    d1,   d2   = df["D_1"].to_numpy(),   df["D_2"].to_numpy()

    # Параметрический тест для возрастов (парный t-test)
    p_age = stats.ttest_rel(age1, age2).pvalue

    # Непараметрический тест для F (если есть разброс)
    p_f = stats.wilcoxon(f1, f2).pvalue if np.var(f1) > 0 else 1.0

    # То же самое для D
    p_d = stats.wilcoxon(d1, d2).pvalue if np.var(d1) > 0 else 1.0

    # Собираем словарь результатов
    return {
        "Pairs": n,
        "p_age": round(p_age, 4), "Age_ns": p_age > alpha,
        "p_f":   round(p_f, 4),   "F_ns":   p_f > alpha,
        "p_d":   round(p_d, 4),   "D_ns":   p_d > alpha
    }


if __name__ == "__main__":
    results = []
    for path in glob("results/pairs_*.csv"):
        stem = os.path.basename(path).removeprefix("pairs_").removesuffix(".csv")
        vaccine, gender = stem.split("_")
        df0 = load_and_clean(path)
        df_matched = build_matched_pairs(df0)
        stats_dict = evaluate_group(df_matched)
        results.append({"Vaccine": vaccine, "Gender": gender, "Total_pairs_initial": df0.height, **stats_dict})

    # Вывод в консоль для проверки
    header = ["Vaccine", "Gender", "Total_pairs_initial", "Pairs", "p_age", "Age_ns", "p_f", "F_ns", "p_d", "D_ns"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for row in results:
        print("| " + " | ".join(str(row[h]) for h in header) + " |")
