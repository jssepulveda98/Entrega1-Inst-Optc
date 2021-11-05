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
	Uf1=np.fft.fftshift(Uf1)

	return Uf1

def secondlens(t2, deltau, deltav, w_length, f_length):
	"""
	Fourier transform done by the second lens
	Incident optical field: plane and monochromatic wave
	Lens bigger than transmittance: pupil function equals 1
	"""

	Uf2=(1/(1j*w_length*f_length))*np.fft.fft2(t2*deltau*deltav)
	Uf2=np.fft.fftshift(Uf2)

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
    
    #B=Window(A,B)
    
    TamA=np.shape(A)
    TAX,TAY=TamA[0],TamA[1]
    TamB=np.shape(B)
    TBX,TBY=TamB[0],TamB[1]
    
    deltau=(0.633*50000)/(2*840*2.99) 
    deltav=(0.633*50000)/(2*1300*2.99)
    u=840*deltau  #plane size of 2*u*2*v
    v=1300*deltav
    
    
    B=firstlens(B,2.99,2.99,0.633,50000)
    
    Autocorr=B*B.conjugate()
    Autocorr=secondlens(Autocorr,deltau,deltav,0.633,50000)
    plt.figure() 
    plt.imshow((np.log(np.abs(Autocorr)**2)), extent=[-u,u,-v,v],cmap="gray")
    plt.title('Auto-correlation')
    plt.ylabel('[um]')
    plt.xlabel('[um]')
   
    for pixelX in np.arange(1,int(TAX -TBX)-10):
        for pixelY in np.arange(1,int(TAY -TBY)-10):
            
            PorA=A[pixelX:TBX+pixelX:1,pixelY:TBY+pixelY:1]
            TransFA=firstlens(PorA,2.99,2.99,0.633,50000)
            c = TransFA*B.conjugate()
            C = secondlens(c,deltau,deltav,0.633,50000)
            
            comparison = abs(C-Autocorr) <=1.5* np.ones(np.shape(C)) 
            print(comparison)
            if comparison.all():
                print(pixelX,pixelY)
                plt.figure() 
                plt.imshow((np.log(np.abs(C)**2)), extent=[-u,u,-v,v],cmap="gray")
                plt.title('Crossed correlation')
                plt.ylabel('[um]')
                plt.xlabel('[um]')
             
    
    
    return C



    
#Image reading

imagen=cv2.imread("c.jpeg",0)
clue=cv2.imread("c_clue.jpeg",0)




Correlate2DF(imagen,clue)
#plot

#plt.imshow(np.log(np.abs(Correlate2DF(imagen,clue))**2),cmap="gray")








plt.show()