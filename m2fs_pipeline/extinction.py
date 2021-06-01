import sys
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt


def curve(x):
    """
    Cardelli 1989
    """
    x = 10000./x
    y = x - 1.82
  
    a=np.polyval([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.], y)
    b=np.polyval([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.], y)
  
    R=3.1
    return a+b/R

def extinction(field, input, output):

    #original sensitivity function
    wave0 = np.genfromtxt(input, comments='#')[:,0]
    factor0 = np.genfromtxt(input, comments='#')[:,1]
    factor0_err = np.genfromtxt(input, comments='#')[:, 2]

    if(field=='GOODSS'):
        Av=0.019
    if(field=='UDS'):
        Av=0.061
    if(field=='COSMOS'):
        Av=0.051

    #A_lambda
    A_lambda = curve(wave0)*Av

    #corrected sensitivity function
    factor=factor0*10**(0.4*A_lambda)
    factor_err = factor0_err*10**(0.4*A_lambda)

#    f=plt.figure()
#    plt.plot(wave0, factor0, label='Original curve')
#    plt.plot(wave0, factor, label='Extinction corrected curve')
#    plt.plot(wave0, factor/factor0)
#    plt.xlabel('Wavelength [Angstroms]')
#    plt.ylabel('(ergs/s/cm2/A)/(e/s/pix)')
#    plt.legend()
#    plt.show()

    lun = open(output, 'w')
    lun.write('# Flux calibration Curve'+'\n')
    lun.write('# Angstroms  (erg/s/cm2/A)/(e/s/pix)'+'\n')
    output_data = np.vstack((wave0, factor, factor_err)).T
    np.savetxt(lun, output_data)
    lun.close()