import cv2

cap = cv2.VideoCapture(0)

# ใช้ loop มา ทำให้เป็นวีดีโอ ไม่ใช่ภาพถ่ายแบบตัวอย่าง 1_1
while(True):
    ret,im = cap.read()

    cv2.imshow('camera',im) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
