import cv2

input("Press the Enter key to continue: ")
video = cv2.VideoCapture(0)
a = 0
while True:
    a = a + 1
    # Create a frame object
    check, frame = video.read()
    # show the frame while capturing the image
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press q to exit
        break

# save image
showPic = cv2.imwrite("filename.jpg", frame)
#print(showPic)
video.release()
