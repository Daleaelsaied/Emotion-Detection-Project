import cv2
from deepface import DeepFace

# فتح الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # تحليل المشاعر
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # لو النتيجة list ناخد أول عنصر
        if isinstance(result, list):
            result = result[0]

        # استخراج العاطفة
        emotion = result['dominant_emotion']

        # كتابة العاطفة على الفيديو
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        print("Analyse error:", e)

    # عرض الفيديو
    cv2.imshow("Emotion Detection", frame)

    # اضغط q للخروج
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
