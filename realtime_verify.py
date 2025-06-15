import cv2
import os
from model.siamese_model import verify, load_model

model = load_model()
input_image_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
os.makedirs(os.path.dirname(input_image_path), exist_ok=True)

cap = cv2.VideoCapture(4)

print("[INFO] Webcam mulai... tekan 'v' untuk verifikasi")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Gagal ambil frame dari webcam")
        break

    cropped = frame[120:120+250, 200:200+250]
    key = cv2.waitKey(1) & 0xFF

    if key == ord('v'):
        print("[INFO] Tombol 'v' ditekan, mulai verifikasi")
        cv2.imwrite(input_image_path, cropped)

        results, verified = verify(model, detection_threshold=0.5, verification_threshold=0.5)
        print('[RESULT]', "✅ VERIFIED" if verified else "❌ UNVERIFIED")

        label = "VERIFIED" if verified else "UNVERIFIED"
        color = (0, 255, 0) if verified else (0, 0, 255)
        overlay = cropped.copy()
        cv2.putText(overlay, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        for _ in range(150):
            cv2.imshow("Verification", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv2.imshow("Verification", cropped)

    if key == ord('q'):
        print("[INFO] Keluar...")
        break

cap.release()
cv2.destroyAllWindows()
