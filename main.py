import cv2

def start_cam(cap):
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not cap.isOpened():
        print("camera is not opened")

def main():
    cap = cv2.VideoCapture(0)

    try:
        start_cam(cap)
    except KeyboardInterrupt:
        print("quitting")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()