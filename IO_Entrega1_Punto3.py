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
	"""
	x=np.arange(-M,M)
	y=np.arange(-N,N)
	x,y=np.meshgrid(x,y)
	lim=5000000
	t2_matrix=(deltau*x)**2 + (deltav*y)**2
	t2_matrix[np.where(t2_matrix<=lim)]=1
	t2_matrix[np.where(t2_matrix>lim)]=0

	t2=t2_matrix*UF1

	I3=(np.abs(t2_matrix)**2) 
	plt.figure(3) 
	plt.imshow(I3, extent=[-u,u,-v,v])
	plt.title('Transmittance in Fourier Plane')
	plt.ylabel('[um]')
	plt.xlabel('[um]')
	plt.imsave("t.png",I3, cmap='gray')                  

	return t2


def secondlens(t2, deltau, deltav, w_length, f_length):
	"""
	Fourier transform done by the second lens
	Incident optical field: plane and monochromatic wave
	Lens bigger than transmittance: pupil function equals 1
	"""

	Uf2=(1/(1j*w_length*f_length))*np.fft.fft2(t2*deltau*deltav)
	#Uf2=np.fft.fftshift(Uf2)

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
deltaxi=pixel size image plane
deltaeta=pixel size image plane
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

#Constrains of pixel sizes do to the Fourier transform discretization
deltau=(w_length*f_length)/(M*deltaxprim) 
deltav=(w_length*f_length)/(N*deltayprim)

deltaxi=(w_length*f_length)/(M*deltau)
deltaeta=(w_length*f_length)/(N*deltav)

#Size of planes
u=M*deltau
v=N*deltav

xi=M*deltaxi
eta=N*deltaeta

#Image formation
t1= cv2.imread("cameraman.png",0)
UF1=firstlens(t1, deltaxprim, deltayprim, w_length, f_length)
t2=transmittanceFP(UF1, w_length, f_length, deltau, deltav, M, N, u, v)
UF2=secondlens(t2, deltau, deltav, w_length, f_length)

#Fourier transform of the image
I1=np.log((np.abs(UF1)**2))                    #Intensity
angle1=np.angle(UF1)                           #Phase

#Output image
I2=(np.abs(UF2)**2)                            #Intensity
angle2=np.angle(UF2)                           #Phase

#Fourier transform of the image multiplied by the trnasmittance t2
I3=(np.abs(t2)**2)                             #Intensity
anglet2=np.angle(t2)                           #Phase


#Plot
plt.figure(1) 
plt.imshow(I1, extent=[-u,u,-v,v])
plt.title('Fourier plane')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("Fourier plane.png",I1, cmap='gray')

plt.figure(2) 
plt.imshow(I2, extent=[-xi,xi,-eta,eta])
plt.title('Image')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("Image.png",I2, cmap='gray')

plt.figure(3) 
plt.imshow(I3, extent=[-u,u,-v,v])
plt.title('Transmittance in Fourier Plane')
plt.ylabel('[um]')
plt.xlabel('[um]')
plt.imsave("t2.png",I3, cmap='gray')