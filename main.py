
import cv2
import dlib
import time
import math


carCascade = cv2.CascadeClassifier('vech.xml')
video = cv2.VideoCapture('test.MOV')

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(
        location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    return d_meters * fps * 3.6


def detectPlates(image):
    plateCascade = cv2.CascadeClassifier(
        'model/haarcascade_russian_plate_number.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return plateCascade.detectMultiScale(gray, 1.1, 4)


def _extracted_from_trackMultipleObjects_35(carID, carTracker, carLocation1, carLocation2):
    print(f"Removing carID {str(carID)} from list of trackers. ")
    print(f"Removing carID {str(carID)} previous location. ")
    print(f"Removing carID {str(carID)} current location. ")
    carTracker.pop(carID, None)
    carLocation1.pop(carID, None)
    carLocation2.pop(carID, None)


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0
    count = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outNew.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker:
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            _extracted_from_trackMultipleObjects_35(
                carID, carTracker, carLocation1, carLocation2
            )
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker:
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print(f' Creating new tracker{str(currentCarID)}')

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(
                        image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker:
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y),
                          (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if end_time != start_time:
            fps = 1.0/(end_time - start_time)

        for i in carLocation1:
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (
                        (speed[i] is None or speed[i] == 0)
                        and y1 >= 275
                        and y1 <= 285
                    ):
                        speed[i] = estimateSpeed(
                            [x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(
                            resultImage,
                            f"{int(speed[i])}km/h",
                            (int(x1 + w1 / 2), int(y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 100),
                            2,
                        )

        plates = detectPlates(resultImage)
        success, img = video.read()
        for (x, y, w, h) in plates:
            area = w * h

            if area > 500:
                cv2.rectangle(resultImage, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resultImage, "Number Plate", (x, y - 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                plate_roi = resultImage[y:y + h, x:x + w]
                cv2.imshow("Plate ROI", plate_roi)

        cv2.putText(
            resultImage,
            f"FPS : {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Result", resultImage)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            cv2.imwrite(f"plates/scaned_img_{str(count)}.jpg", plate_roi)
            cv2.rectangle(plate_roi, (0, 200), (640, 300),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(plate_roi, "Plate Saved", (150, 265),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Results", plate_roi)
            cv2.waitKey(500)
            count += 1

    cv2.destroyAllWindows()


trackMultipleObjects()
