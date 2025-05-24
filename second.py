import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def uploadFiles():
    # бд без моно
    data = pd.read_excel("results/full_BD_isAlive.xlsx")
    #print(data.shape)
    #print(data.info())
    return data

if __name__ == "__main__":
    # считываем
    data = uploadFiles()

    # определяем какие вакцины использовались
    vacNames = set(data["Vac"])
    vacNames = list(vacNames)

    # создаем свой дф для каждой вакцины
    vacFrames = []
    for name in vacNames:
        newDf = data[data["Vac"] == name]
        vacFrames.append(newDf)

    # сохраняем данные по каждой вакцине в файл (для след пунктов задания)
    vacNum = 0
    for name in vacNames:
        vacFrames[vacNum].to_excel('results/'+name+'.xlsx')
        vacNum += 1

    # вычисление и вывод данных
    fig, ax = plt.subplots(2,3,figsize=(14, 28))
    for i in range(2):
        for j in range(3):
            ax[i,j].axis('off')
    

    i = 0
    j = 0
    vacNum = 0
    for name in vacNames:
        
        # количество мужчин/женщин
        male = len(vacFrames[vacNum][vacFrames[vacNum]['Gender'] == 'м'])
        female = len(vacFrames[vacNum][vacFrames[vacNum]['Gender'] == 'ж'])
        all = male+female
        # средний возраст
        midAge = np.ceil(vacFrames[vacNum]['Age'].mean())
        # количество выживших
        alive = len(vacFrames[vacNum][vacFrames[vacNum]['Outcome'] == 'Выписан'])
        dead = len(vacFrames[vacNum][vacFrames[vacNum]['Outcome'] == 'Умер'])

        # вложенные оси для данных
        ax0 = ax[i,j].inset_axes([0, 1, 1, 0.1])
        ax1 = ax[i,j].inset_axes([0, 0.9, 1, 0.1])
        ax2 = ax[i,j].inset_axes([0, 0.5, 1, 0.4])
        ax3 = ax[i,j].inset_axes([0, 0, 1, 0.4])

        # вывод данных
        ax0.text(0.4,0,name, fontsize='xx-large',fontweight='bold')
        ax1.text(0, 0, 'Средний возраст: ' + str(midAge), fontsize='x-large')
        ax2.pie([male, female], labels=['м','ж'],autopct='%1.1f%%')
        ax3.pie([alive, dead], labels=['выписан','умер'],autopct='%1.1f%%')
        
        ax0.axis('off')
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')

        j = (j+1)%3
        if j==2:
            i+=1
        vacNum+=1
    
    plt.show()