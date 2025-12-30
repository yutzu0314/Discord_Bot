import cv2

#url = "https://tcnvr3.taichung.gov.tw/f3949e40"
#url = "rtmp://skyeyes.thu.edu.tw/live/yuenonglixin1"
url = "rtsp://rtsp:rtsp1234@140.128.124.58:7177/cam/realmonitor?channel=24&subtype=0"
cap = cv2.VideoCapture(url)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"你點的座標：({x}, {y})")

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("讀不到畫面")
        break

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
