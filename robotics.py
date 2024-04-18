import cv2
import numpy as np

image_path = "C:/Users/pundr/Downloads/WhatsApp Image 2024-04-04 at 11..jpg"


use_camera = True


video_path = "C:/Users/pundr/Downloads/WhatsApp Video 2024-04-18 at 10.42.09 AM.mp4"


def define_color_range(image):
    """
    Defines color range for object detection based on a sample image.

    Args:
        image: Path to the image containing the object of interest.

    Returns:
        A tuple containing the lower and upper bounds for the HSV color range.
    """

    image = cv2.imread(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    roi = cv2.selectROI("Select Object Color", hsv, False)
    x1, y1, w, h = roi


    roi_hsv = hsv[y1:y1+h, x1:x1+w]

    avg_hsv = np.average(roi_hsv, axis=0).astype(np.uint8)


    lower_color = np.array([avg_hsv[0] - 10, avg_hsv[1] - 30, avg_hsv[2] - 30])
    upper_color = np.array([avg_hsv[0] + 10, avg_hsv[1] + 30, avg_hsv[2] + 30])

    return lower_color, upper_color


def main():

    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)


    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    canvas = None


    if not use_camera:
        define_color_range(image_path)

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)


        if canvas is None:
            canvas = np.zeros_like(frame)


        cv2.imshow('image2', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
