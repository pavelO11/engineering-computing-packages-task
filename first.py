import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def uploadFiles():
    # бд без моно
    bdNonMono = pd.read_excel("data/БД_без_моно_full.xlsx")
    #print("\n\nбд без моно:\n")
    #print(bdNonMono.shape)
    #print(bdNonMono.info())
    #print(bdNonMono.iloc[74])

    # бд с моно
    bdMono = pd.read_excel("data/БД_с_моно_full.xlsx")
    #print("\n\nбд с моно:\n")
    #print(bdMono.shape)
    #print(bdMono.info())

    # показатель D
    Dindex = pd.read_excel("data/Показатель_D.xlsx")
    #print("\n\nпоказатель D:\n")
    #print(Dindex.shape)
    #print(Dindex.info())

    # показатель F
    Findex = pd.read_excel("data/Показатель_F.xlsx")
    #print("\n\nпоказатель F:\n")
    #print(Findex.shape)
    #print(Findex.info())

    return bdNonMono, bdMono, Dindex, Findex

def generateID(existing, min_val=1, max_val=1000):
    while True:
        num = random.randint(min_val, max_val)
        if num not in existing:
            existing.add(num)  # Добавляем номер, чтобы избежать повторов
            return num

if __name__ == "__main__":
    # считываем
    bdNonMono, bdMono, Dindex, Findex = uploadFiles()

    # дополняем nan номера уникальными значениями
    CaseIds = set(bdNonMono["CaseID"]) | set(bdMono["CaseID"]) | set(Findex["CaseID"]) | set(Dindex["CaseID"])
    bdNonMono.loc[bdNonMono['CaseID'].isna(), 'CaseID'] = generateID(CaseIds, 0, max(CaseIds)+1000)
    bdMono.loc[bdMono['CaseID'].isna(), 'CaseID'] = generateID(CaseIds, 0, max(CaseIds)+1000)
    Findex.loc[Findex['CaseID'].isna(), 'CaseID'] = generateID(CaseIds, 0, max(CaseIds)+1000)
    Dindex.loc[Dindex['CaseID'].isna(), 'CaseID'] = generateID(CaseIds, 0, max(CaseIds)+1000)
    
    # удаляем дубли
    bdNonMono.drop_duplicates(keep='last', inplace=True)
    bdMono.drop_duplicates(keep='last', inplace=True)
    Dindex.drop_duplicates(keep='last', inplace=True)
    Findex.drop_duplicates(keep='last', inplace=True)

    # Объединяем с порядком и именами столбцов как в бд_без_моно
    columns_order = ['CaseID', 'Start', 'End', 'Gender', 'Age', 'Ther', 'Outcome', 'Vac']
    
    bdMono = bdMono.rename(columns={'Vacin': 'Vac'})
    bdMono = bdNonMono[columns_order]

    fullBD = pd.concat([bdNonMono, bdMono])

    # проверяем на пустые столбцы и забиваем дефолт значением
    
    # после выполнения этого цикла видим, что пустые есть только в столбце с вакцинацией
    '''
    for c in columns_order:
        print(c + ' ' + str(sum(fullBD[c].isna())))
    '''
    fullBD = fullBD.fillna('Нет')

    # оставляем только записи с известным исходом жив/мертв
    fullBD = fullBD.drop(fullBD[fullBD['Outcome'] == 'Переведён в др. ЛПУ'].index)

    # переводим dataframe обратно в файл
    fullBD.to_excel('results/full_BD_isAlive.xlsx') 
