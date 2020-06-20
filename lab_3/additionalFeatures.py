import cv2

bins = 8
def fd_Fast(_image):
    imag = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(gray, None)
    imag2 = cv2.drawKeypoints(gray, kp, None, color=(255, 0, 0))
    cv2.normalize(imag2, imag2)

    return imag2.flatten()

def fd_Kaze(_image):
    alg = cv2.KAZE_create()
    # Dinding image keypoints
    kps = alg.detect(_image)
    # Getting first 32 of them.
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:32]
    # computing descriptors vector
    kps, dsc = alg.compute(_image, kps)
    cv2.normalize(dsc, dsc)
    # Flatten all of them in one big vector - our feature vector
    return dsc.flatten()