import numpy as np, cv2

def doCorrectionImage(image, face_center, eye_centers):

    pt0, pt1 = eye_centers

    if pt0[0] > pt1[0]: pt0, pt1 = pt1, pt0

    dx, dy = np.subtract(pt1, pt0).astype(float)

    angle = cv2.fastAtan2(dy, dx)

    rot = cv2.getRotationMatrix2D(face_center, angle, 1)

    size = image.shape[1::-1]

    correction_image = cv2.warpAffine(image, rot, size, cv2.INTER_CUBIC)

    eye_centers = np.expand_dims(eye_centers, axis=0)
    correction_centers = cv2.transform(eye_centers, rot)
    correction_centers = np.squeeze(correction_centers, axis=0)

    return correction_image, correction_centers