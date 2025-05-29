import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg
import second

def button_click_P2(event):
    second.main(False)

def button_click_P5(event):
    img1 = mpimg.imread('table_images/isAvac.jpg')
    img2 = mpimg.imread('table_images/lengthAll.jpg')
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].axis('off')
    ax[0].imshow(img1)
    ax[1].axis('off')
    ax[1].imshow(img2)
    plt.show()

def button_click_Age(event):
    img1 = mpimg.imread('table_images/AgeRes.jpg')
    img2 = mpimg.imread('table_images/AgeLen.jpg')
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].axis('off')
    ax[0].imshow(img1)
    ax[1].axis('off')
    ax[1].imshow(img2)
    plt.show()

def button_click_Vac(event):
    img1 = mpimg.imread('table_images/VacRes.jpg')
    img2 = mpimg.imread('table_images/VacLen.jpg')
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].axis('off')
    ax[0].imshow(img1)
    ax[1].axis('off')
    ax[1].imshow(img2)
    plt.show()

def button_click_Sex(event):
    img1 = mpimg.imread('table_images/SexRes.jpg')
    img2 = mpimg.imread('table_images/SexLen.jpg')
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].axis('off')
    ax[0].imshow(img1)
    ax[1].axis('off')
    ax[1].imshow(img2)
    plt.show()

def button_click_Taj(event):
    img1 = mpimg.imread('table_images/TajRes.jpg')
    img2 = mpimg.imread('table_images/TajLen.jpg')
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].axis('off')
    ax[0].imshow(img1)
    ax[1].axis('off')
    ax[1].imshow(img2)
    plt.show()


def button_click_Dind(event):
    img1 = mpimg.imread('table_images/Dres.jpg')
    img2 = mpimg.imread('table_images/DLen.jpg')
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].axis('off')
    ax[0].imshow(img1)
    ax[1].axis('off')
    ax[1].imshow(img2)
    plt.show()


def button_click_Find(event):
    img1 = mpimg.imread('table_images/FRes.jpg')
    img2 = mpimg.imread('table_images/FLen.jpg')
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].axis('off')
    ax[0].imshow(img1)
    ax[1].axis('off')
    ax[1].imshow(img2)
    plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(13, 15))
    ax.axis('off')

    button_ax = plt.axes([0.2, 0.877, 0.6, 0.1])
    buttonP2 = Button(button_ax, 'Описательная статистика по вакцинам', color='lightgray', hovercolor='lightgreen')
    button_ax = plt.axes([0.2, 0.755, 0.6, 0.1])
    buttonP5 = Button(button_ax, 'Сравнение групп по исходам', color='lightgray', hovercolor='lightgreen')
    button_ax = plt.axes([0.2, 0.633, 0.6, 0.1])
    buttonAge = Button(button_ax, 'Влияние возраста', color='lightgray', hovercolor='lightgreen')
    button_ax = plt.axes([0.2, 0.511, 0.6, 0.1])
    buttonVac = Button(button_ax, 'Влияние вакцинации', color='lightgray', hovercolor='lightgreen')
    button_ax = plt.axes([0.2, 0.388, 0.6, 0.1])
    buttonSex = Button(button_ax, 'Влияние Пола', color='lightgray', hovercolor='lightgreen')
    button_ax = plt.axes([0.2, 0.266, 0.6, 0.1])
    buttonTaj = Button(button_ax, 'Влияние тяжести состояния', color='lightgray', hovercolor='lightgreen')
    button_ax = plt.axes([0.2, 0.144, 0.6, 0.1])
    buttonDind = Button(button_ax, 'Влияние D-димер', color='lightgray', hovercolor='lightgreen')
    button_ax = plt.axes([0.2, 0.022, 0.6, 0.1])
    buttonFind = Button(button_ax, 'Влияние Ферритин', color='lightgray', hovercolor='lightgreen')

    buttonP2.on_clicked(button_click_P2)
    buttonP5.on_clicked(button_click_P5)
    buttonAge.on_clicked(button_click_Age)
    buttonVac.on_clicked(button_click_Vac)
    buttonSex.on_clicked(button_click_Sex)
    buttonTaj.on_clicked(button_click_Taj)
    buttonDind.on_clicked(button_click_Dind)
    buttonFind.on_clicked(button_click_Find)

    plt.show()
