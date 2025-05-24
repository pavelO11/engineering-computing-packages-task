#pip install polars
import polars as pl

# Читаем файлы (csv)
bd_without_mono = pl.read_csv("БД_без_моно_full_byPavel.csv", encoding="utf-8", separator=",")
bd_with_mono = pl.read_csv("БД_с_моно_full_byPavel.csv", encoding="utf-8", separator=",")
# rider_D = pl.read_csv("Показатель_D_byPavel.csv", encoding="utf-8", separator=";")
# rider_F = pl.read_csv("Показатель_F_byPavel.csv", encoding="utf-8", separator=";")

# Объединяем два DataFrame'а по строкам
df = pl.concat([bd_without_mono, bd_with_mono])


