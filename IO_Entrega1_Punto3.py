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

def secondlens(Uf1, deltau, deltav, w_length, f_length):
	"""
	Fourier transform done by the first lens
	Incident optical field: plane and monochromatic wave
	Lens bigger than transmittance: pupil function equals 1
	"""

	Uf2=(1/(1j*w_length*f_length))*np.fft.fft2(Uf1*deltau*deltav)
	#Uf2=np.fft.fftshift(Uf2)

	return Uf2



#Dimensions and constants
"""
t1=transmittance
f_length=focal length
w_length= wavelength
deltau=pixel size Fourier plane
deltav=pixel size Fourier plane
deltaxprim=pixel size transmittance plane
deltayprim=pixel size transmittance plane
M*2=number of pixels in the x' axis
N*2=number of pixels in the y' axis
M*2xN*2=number of pixels entrance plane
"""

M=256 #Number of pixels=M*2 (along one axis number of pixels=512)
N=256 #Number of pixels=N*2
f_length=50000  #(50mm)
w_length=0.633   #All units in um
deltaxprim=2.99 
deltayprim=2.99

deltau=(w_length*f_length)/(M*deltaxprim)
deltav=(w_length*f_length)/(M*deltayprim)

deltaxi=(w_length*f_length)/(M*deltau)
deltaeta=(w_length*f_length)/(M*deltav)

t1= cv2.imread("cameraman.png",0)
UF1=firstlens(t1, deltaxprim, deltayprim, w_length, f_length)

UF2=secondlens(UF1, deltau, deltav, w_length, f_length)

I1=np.log((np.abs(UF1)**2))                             #Intensity
angle1=np.angle(UF1)                           #Phase

I2=(np.abs(UF2)**2)                            #Intensity
angle2=np.angle(UF2)                           #Phase

u=M*deltau
v=N*deltav

xi=M*deltaxi
eta=N*deltaeta

plt.figure(1) 
plt.imshow(I1)
plt.title('Fourier plane')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("Fourier plane.png",I1, cmap='gray')

plt.figure(2) 
plt.imshow(I2)
plt.title('Image')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("Image.png",I2, cmap='gray')