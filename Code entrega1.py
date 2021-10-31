import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

#Dimensiones y constantes
ABC=np.loadtxt('a.txt',delimiter=',', dtype=complex)

A=[]

print(ABC[0:1])

ABC=np.fft.ifft(ABC)


#Funciones

def f4G(Imagen):
    Imagenf=np.fft.fft2(Imagen)
    filtro=gaussian_filter(Imagen,2.8)
    Imagenff=np.fft.ifft2(Imagenf)
    
    
    return filtro



    
imagen= cv2.imread("b.png")

#plt.imshow(ABC)
#plt.imshow(f4G(imagen))


plt.show()








