from __future__ import annotations
from scipy.signal import convolve2d
import numpy as np

HORN_SCHUNCK_KERNEL = np.array([
    [1 / 12, 1 / 6, 1 / 12], 
    [1 / 6, 0, 1 / 6], 
    [1 / 12, 1 / 6, 1 / 12]], float
)


kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx

kernelY = np.array([[-1, -1], [1, 1]]) * 0.25  # kernel for computing d/dy

kernelT = np.ones((2, 2)) * 0.25


def computeDerivatives(
    im1: np.ndarray, im2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    fx = convolve2d(im1, kernelX, "same") + convolve2d(im2, kernelX, "same")
    fy = convolve2d(im1, kernelY, "same") + convolve2d(im2, kernelY, "same")

    # ft = im2 - im1
    ft = convolve2d(im1, kernelT, "same") + convolve2d(im2, -kernelT, "same")

    return fx, fy, ft

def HornSchunck(
    im1: np.ndarray, 
    im2: np.ndarray, *,
    alpha: float=0.001, 
    iterations: int=8) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------

    im1: numpy.ndarray
        image at t=0
    im2: numpy.ndarray
        image at t=1
    alpha: float
        regularization constant
    Niter: int
        number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)
    vInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    '''
    if verbose:
        from .plots import plotderiv
        plotderiv(fx, fy, ft)
    '''

    # Iteration to reduce error
    for _ in range(iterations):
        
        # %% Compute local averages of the flow vectors
        uAvg = convolve2d(U, HORN_SCHUNCK_KERNEL, "same")
        vAvg = convolve2d(V, HORN_SCHUNCK_KERNEL, "same")

        # %% common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        
        # %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


