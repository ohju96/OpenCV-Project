import cv2

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

cam = cv2.VideoCapture("movie/bts.mp4")

while True:
    ret, movie_image = cam.read()

    if ret is True:
        gray = cv2.cvtColor(movie_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.5, 5, 0, (20, 20))

        for face in faces:
            x, y, w, h = face

            face_image = movie_image[y:y + h, x:x + w]

            cv2.rectangle(movie_image, face, (255, 0, 0), 4)

        cv2.imshow("movie_face", movie_image)

    if cv2.waitKey(1) > 0:
        break