import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureController:
    def __init__(self, screen_width, screen_height):
        self.cap = cv2.VideoCapture(0)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.enabled = True

        # Initialize the new Tasks API
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def get_target_position(self):
        if not self.enabled:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        target = None
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            tip = hand[8]  # Index finger tip
            target = (int(tip.x * self.screen_width),
                      int(tip.y * self.screen_height))
            
            # Draw landmarks manually
            for lm in hand:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(frame, "Press 'G' in Pygame window to toggle | 'Q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Gesture Control - DVRP-MHSI", frame)
        cv2.waitKey(1)
        return target

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()