import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
    # Input:
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the car
    #   (top left, bot right coordinates)
    #   p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #   p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p = p0
    #normalize It, It1
    It = It/max(It.flatten())
    It1 = It1/max(It1.flatten())
    H, W = It1.shape
    print(It.shape)
    # set the threshold
    threshold = 0.01
    # since rect is float, cast to int
    y1, x1, y2, x2 = np.around(rect)
    # get the template
    T_It = It[int(y1):int(y2)+1,int(x1):int(x2)+1]
    # get the gradiaent
    d_x, d_y = np.gradient(It1)
    del_p = np.inf
    jacobian = np.array([[1,0],[0,1]])

    It1_spline = RectBivariateSpline(np.arange(H), np.arange(W), It1)
    dx_spline = RectBivariateSpline(np.arange(H), np.arange(W), d_x)
    dy_spline = RectBivariateSpline(np.arange(H), np.arange(W), d_y)

    x_warp, y_warp = np.meshgrid(
        np.arange(x1, x2+1), np.arange(y1, y2+1))


    while np.linalg.norm(del_p) >= threshold:
        W_It1 = It1_spline.ev(y_warp+p[0], x_warp+p[1])
        b = T_It - W_It1
        b = b.flatten()
        d_x_It1 = dx_spline.ev(y_warp+p[0], x_warp+p[1]).flatten()
        d_y_It1 = dy_spline.ev(y_warp+p[0], x_warp+p[1]).flatten()
        A = np.vstack((d_x_It1, d_y_It1))
        A = A.T
        H = A.T.dot(A)
        invH = np.linalg.inv(H)
        del_p = invH.dot(A.T).dot(b)
        p = p+del_p
    print('p ', p)
    return p
