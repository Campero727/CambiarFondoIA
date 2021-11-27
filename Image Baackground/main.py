import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation=mp.solutions.selfie_segmentation

cap=cv2.VideoCapture(1,cv2.CAP_DSHOW)

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
    while True:
        ret,frame=cap.read()
        if(ret==False):
            break

        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=selfie_segmentation.process(frame_rgb)

        _,th=cv2.threshold(results.segmentation_mask,0.75,255,cv2.THRESH_BINARY)

        th=th.astype(np.uint8)
        th=cv2.medianBlur(th,13)
        th_inv=cv2.bitwise_not(th);

        bg_image=cv2.imread("fondo1.jpg")
        bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        bg=cv2.GaussianBlur(bg_image,(15,15),0);
        bg = cv2.bitwise_and(bg_image, bg_image, mask=th_inv);

        fg=cv2.bitwise_and(frame,frame,mask=th);

        output_image=cv2.add(bg,fg);

        cv2.imshow("output_image", output_image);
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF==27:
            break

cap.release();
cv2.destroyAllWindows()