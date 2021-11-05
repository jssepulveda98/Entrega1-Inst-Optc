import matplotlib.pyplot as plt
import numpy as np
import cv2


#Functions

def firstlens(t1, deltaxprim, deltayprim, w_length, f_length):
	"""
	Fourier transform done by the first lens
	Incident optical field: plane and monochromatic wave
	Lens bigger than transmittance: pupil function equals 1
	"""

	Uf1=(1/(1j*w_length*f_length))*np.fft.fft2(t1*deltaxprim*deltayprim)
	Uf1=np.fft.fftshift(Uf1)

	return Uf1


def transmittanceFP(UF1, w_length, f_length, deltau, deltav, M, N, u, v):
	"""
	Transmittance in the Fourier Plane to manipulate the frequencies of the image
	In this case to filter out the frequencies corresponding to the lines of the image
	"""
	x=np.arange(-M,M)
	y=np.arange(-N,N)
	x,y=np.meshgrid(x,y)
	lim=500**2   #radius of 500um
	a=13*deltau  #position of object
	b=122*deltav
	Obj1=(deltau*x+a)**2 + (deltav*y+b)**2
	Obj1[np.where(Obj1<=lim)]=0
	Obj1[np.where(Obj1>lim)]=1

	Obj2=(deltau*x-a)**2 + (deltav*y-b)**2
	Obj2[np.where(Obj2<=lim)]=0
	Obj2[np.where(Obj2>lim)]=1


	t2=Obj1*Obj2*UF1          

	return t2,Obj1,Obj2


def secondlens(t2, deltau, deltav, w_length, f_length):
	"""
	Fourier transform done by the second lens
	Incident optical field: plane and monochromatic wave
	Lens bigger than transmittance: pupil function equals 1
	"""

	Uf2=(1/(1j*w_length*f_length))*np.fft.fft2(t2*deltau*deltav)

	return Uf2


#Dimensions and constants
"""
t1=transmittance in the ingoing focal plane of the first lens (image)
f_length=focal length
w_length= wavelength
deltau=pixel size Fourier plane
deltav=pixel size Fourier plane
deltaxprim=pixel size transmittance (t1) plane
deltayprim=pixel size transmittance (t1) plane
deltaxi=pixel size image plane (output)
deltaeta=pixel size image plane (output)
M*2=number of pixels in the x' axis
N*2=number of pixels in the y' axis
M*2xN*2=number of pixels entrance plane
"""

M=384 #Number of pixels=M*2 (number of pixels along one axis=768)
N=384 #Number of pixels=N*2
f_length=50000  #(50mm)
w_length=0.633  #(633nm orange/red) #All units in um
deltaxprim=2.5 
deltayprim=2.5

#Constrains of pixel sizes do to the Fourier transform discretization
deltau=(w_length*f_length)/(2*M*deltaxprim) 
deltav=(w_length*f_length)/(2*N*deltayprim)

deltaxi=(w_length*f_length)/(2*M*deltau)
deltaeta=(w_length*f_length)/(2*N*deltav)

#Size of planes
u=M*deltau  #plane size of 2*u*2*v
v=N*deltav

xi=M*deltaxi
eta=N*deltaeta


#Image formation
t1= cv2.imread("b.png",0)
UF1=firstlens(t1, deltaxprim, deltayprim, w_length, f_length)
t2,Obj1,Obj2=transmittanceFP(UF1, w_length, f_length, deltau, deltav, M, N, u, v)
UF2=secondlens(t2, deltau, deltav, w_length, f_length)

#Fourier transform of the image
I1=np.log((np.abs(UF1)**2))                    #Intensity
angle1=np.angle(UF1)                           #Phase

#Output image
I2=(np.abs(UF2)**2)                            #Intensity
angle2=np.angle(UF2)                           #Phase

#Transmittance t2
I3=(np.abs(Obj1)**2)                      #Intensity

#Transmittance t2
I4=(np.abs(Obj2)**2)                      #Intensity

#Fourier transform of the image multiplied by the transmittance t2 
I5=I3*I4*I1                                       #Intensity


#Plot
#Fourier Transform of image
plt.figure(1) 
plt.imshow(I1, extent=[-u,u,-v,v])
plt.title('Fourier plane')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("Fourier planeP4b.png",I1, cmap='gray')

#Output image
plt.figure(2) 
plt.imshow(I2, extent=[-xi,xi,-eta,eta])
plt.title('Output image')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("Output imageP4b.png",I2, cmap='gray')

#Transmittance t2
plt.figure(3) 
plt.imshow(I3, extent=[-u,u,-v,v])
plt.title('Transmittance in Fourier Plane')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("filterP4b.png",I3, cmap='gray')    

#Transmittance t2
plt.figure(3) 
plt.imshow(I4, extent=[-u,u,-v,v])
plt.title('Transmittance in Fourier Plane')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("filter2P4b.png",I4, cmap='gray')   

#Fourier transform of the image multiplied by the transmittance t2
plt.figure(4) 
plt.imshow(I5, extent=[-u,u,-v,v])
plt.title('Transmittance in Fourier Plane')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("FTfilteredP4b.png",I5, cmap='gray')   