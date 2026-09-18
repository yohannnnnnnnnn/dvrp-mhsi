import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera open nahi ho raha!")
else:
    print("✅ Camera mil gaya! Window khul rahi hai...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame nahi aa raha")
            break
        cv2.imshow("Webcam Test - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()