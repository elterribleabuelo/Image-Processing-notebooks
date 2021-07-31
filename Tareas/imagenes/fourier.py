import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('J1.png',0)

# =============================================================================
# TRANSFORMADA DE FOURIER
# =============================================================================
f = np.fft.fft2(img)

# =============================================================================
# FFTSHIFT
# =============================================================================
fshift = np.fft.fftshift(f)
Fourier = np.fft.fftshift(f)


### MODULO Y ESCALA LOGARITMICA DE LA TF ###
Fourier = np.log(np.abs(Fourier) + 1)

# =============================================================================
# Desarrollo del filtro
# =============================================================================
#rows, cols = img.shape
#crow,ccol = int(rows/2) , int(cols/2)
fshift[425:445,330:350] = 0
fshift[355:375,455:475] = 0
fshift[505:520,230:250] = 0  
fshift[290:305,560:580] = 0
#fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

### MODULO Y ESCALA LOGARITMICA DEL FILTRO ###
filtro = np.log(np.abs(fshift) + 1)

# =============================================================================
# Transformada inversa de Fourier
# =============================================================================
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

img_back.max()

def Normalizar(f):
  faux = np.ravel(f).astype(float)
  minimum = faux.min()
  maximum = faux.max()
  g = (faux-minimum)*(255) / (maximum-minimum)
  r = g.reshape(f.shape).astype(np.uint8)
  return(r)
# =============================================================================
# Visualizacion
# =============================================================================


cv2.imshow("img",img)
cv2.imshow("img_back",Normalizar(img_back))
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(142),plt.imshow(Fourier, cmap = 'gray')
plt.title('Fourier'), plt.xticks([]), plt.yticks([])


plt.subplot(143),plt.imshow(filtro, cmap = 'gray')
plt.title('Filtro'), plt.xticks([]), plt.yticks([])


plt.subplot(144),plt.imshow(img_back,cmap = 'gray')
plt.title('Transformada inversa'), plt.xticks([]), plt.yticks([])

plt.show()

