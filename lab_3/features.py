import cv2
import mahotas

bins = 8

def fd_4(_image, mask=None):
    _imag = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
    _gray = cv2.cvtColor(_imag, cv2.COLOR_BGR2GRAY)
    _sift = cv2.xfeatures2d_SIFT.create()
    _KP = _sift.detect(_gray, None)
    _imag = cv2.drawKeypoints(_gray, _KP, _imag)
    cv2.normalize(_imag, _imag)

    return _imag.flatten()

def fd_haralick(_image):
    _gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    _haralick = mahotas.features.haralick(_gray).mean(axis=0)

    return _haralick

def fd_hu_moments(_image):
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    _feature = cv2.HuMoments(cv2.moments(_image)).flatten()

    return _feature

def fd_histogram(_image, mask=None):
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
    _hist = cv2.calcHist([_image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(_hist, _hist)

    return _hist.flatten()
