from __future__ import annotations
from scipy.signal import convolve2d
import numpy

# The Horn Schunck kernel 
HORN_SCHUNCK_KERNEL = numpy.array([
    [1 / 12, 1 / 6, 1 / 12], 
    [1 / 6, 0, 1 / 6], 
    [1 / 12, 1 / 6, 1 / 12]], float
)

# A kernel for computing d/dx
KERNEL_DX = numpy.array([[-1, 1], [-1, 1]]) * 0.25  

# A kernel for computing d/dy
KERNEL_DY = numpy.array([[-1, -1], [1, 1]]) * 0.25  

# Computing the differences between frames 
KERNEL_DT = numpy.ones((2, 2)) * 0.25


####################################################################################################
# @compute_derivatives
####################################################################################################
def compute_derivatives(frame1: numpy.ndarray, 
                        frame2: numpy.ndarray) -> \
                        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:


    # Compute the derivatives along X and Y
    fx = convolve2d(frame1, KERNEL_DX, "same") + convolve2d(frame2, KERNEL_DX, "same")
    fy = convolve2d(frame1, KERNEL_DY, "same") + convolve2d(frame2, KERNEL_DY, "same")

    # Compute the derivative along the time ft = frame1 - frame2
    ft = convolve2d(frame1, KERNEL_DT, "same") + convolve2d(frame2, -KERNEL_DT, "same")

    return fx, fy, ft


####################################################################################################
# @compute_optical_flow
####################################################################################################
def compute_optical_flow(frame1: numpy.ndarray, 
                         frame2: numpy.ndarray, *,
                         alpha: float=0.001, 
                         iterations: int=8) -> \
                         tuple[numpy.ndarray, numpy.ndarray]:

    # Ensure that the frames are of type float 32
    frame1 = frame1.astype(numpy.float32)
    frame2 = frame2.astype(numpy.float32)

    # Set up initial flow fields
    uInitial = numpy.zeros([frame1.shape[0], frame1.shape[1]], dtype=numpy.float32)
    vInitial = numpy.zeros([frame1.shape[0], frame1.shape[1]], dtype=numpy.float32)

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = compute_derivatives(frame1, frame2)

    # Iteration to reduce error
    for i in range(iterations):
        
        # Compute local averages of the flow fields (vectors)
        uAvg = convolve2d(U, HORN_SCHUNCK_KERNEL, "same")
        vAvg = convolve2d(V, HORN_SCHUNCK_KERNEL, "same")

        # Common part of the update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        
        # Apply an iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    # Return the optical flow field as separate numpy arrays along the X and Y directions 
    return U, V
