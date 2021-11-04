import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssl
import cv2

def firstlens(t1, deltaxprim, deltayprim, w_length, f_length):
	"""
	Fourier transform done by the first lens
	Incident optical field: plane and monochromatic wave
	Lens bigger than transmittance: pupil function equals 1
	"""

	Uf1=(1/(1j*w_length*f_length))*np.fft.fft2(t1*deltaxprim*deltayprim)
	#Uf1=np.fft.fftshift(Uf1)

	return Uf1

def secondlens(t2, deltau, deltav, w_length, f_length):
	"""
	Fourier transform done by the second lens
	Incident optical field: plane and monochromatic wave
	Lens bigger than transmittance: pupil function equals 1
	"""

	Uf2=(1/(1j*w_length*f_length))*np.fft.fft2(t2*deltau*deltav)
	#Uf2=np.fft.fftshift(Uf2)

	return Uf2

   
def Window(Imagen,Clue):
    """
    Creating a window to get the same pixelsize for both images, the 
    Clue is ubicated in the middle
    
    """
    
    TamIm=np.shape(Imagen)
    XIm, YIm=TamIm[0],TamIm[1]
    TamClue=np.shape(Clue)
    XC, YC=TamClue[0],TamClue[1]
    Window=np.zeros(TamIm)
    Window[int((XIm/2)-(XC/2) +1):int((XIm/2) +(XC/2) +1),int((YIm/2)-(YC/2) +1):int((YIm/2) +(YC/2) +1)]=Clue
    
    return Window
    


def Correlate2DF(A,B):
    """
    ---
    A : Image
    B : Clue
    Returns
    -------
    C : Crossed Correlation matrix

    """
    
    B=Window(A,B)
    
    A=firstlens(A,2.99,2.99,0.633,50000)
    B=firstlens(B,2.99,2.99,0.633,50000)
    
    c = A*B.conjugate()
    C = secondlens(c,2.99,2.99,0.633,50000)
    C = np.fft.fftshift(C)
    
    return C


def LocateWaldo( orig, clue ):
    """
    Search for the maximum point of correlation 

    """
    corr = Correlate2DF(orig, clue)
    V,H = corr.shape
    v,h = divmod( abs(corr).argmax(), H )
    return v,h


def Mask(Orig,Clue):
    """
    Generates a mask in the pointof maximun correlation in order to enhance Waldo
    
    """
    
    coord=LocateWaldo(Orig,Clue)
    Xce,Yce=coord[0],coord[1]
    TamC=np.shape(Clue)
    Xc,Yc=TamC[0],TamC[1]
    Mask=np.zeros(np.shape(Orig))
    Mask[int(Xce-(Xc/2)+1):int(Xce+(Xc/2)+1),int(Yce-(Yc/2)+1):int(Yce+(Yc/2)+1)]=200*np.ones(TamC)
    
    return Mask 
    
#Image reading

imagen=cv2.imread("c.jpeg",0)
clue=cv2.imread("c_clue.jpeg",0)





#plot

plt.imshow(imagen -Mask(imagen,clue),cmap="gray")








plt.show()