import numpy as np

from lick import lick


def test_single_prec():
    x = np.linspace(0.1, 10, 2048)
    y = np.linspace(0.1, 5, 1024)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    V1 = np.cos(XX).astype("float32")
    V2 = np.sin(YY).astype("float32")

    lick(V1, V2, niter_lic=1)
