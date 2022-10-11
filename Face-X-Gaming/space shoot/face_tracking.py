import cv2
import utils
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

BLUE = (255, 0, 0, 0)
GREEN = (0, 255, 0, 0)
RED = (0, 0, 255, 0)
NOSE = [4]

TOP_LEFT_X = 220
TOP_LEFT_Y = 50
BOTTOM_RIGHT_X = int(utils.SCREEN_WIDTH/2) + TOP_LEFT_X
BOTTOM_RIGHT_Y = int(utils.SCREEN_HEIGHT/2) + TOP_LEFT_Y


class FaceTracker:
    def __init__(self):
        self.game_pos = {}

    def get_landmark_pos(self, face_mesh_results_multi_face_landmarks, flm_list, image):
        h, w, c = image.shape

        for face_landmarks in face_mesh_results_multi_face_landmarks:
            for idd, lm in enumerate(face_landmarks.landmark):
                if idd in flm_list:
                    x, y = int(lm.x * w), int(lm.y * h)
                    pos = {"idd": idd, "x": x, "y": y}
                    return pos
        
    def convert_landmark_pos_to_game_pos(self, pos):
        self.game_pos["x"] = (BOTTOM_RIGHT_X - pos['x']) * 2
        self.game_pos["y"] = (pos['y'] - TOP_LEFT_Y) * 2

                    
    def draw_landmark(self, pos, image):
        cv2.rectangle(image, (TOP_LEFT_X, TOP_LEFT_Y), 
                (BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y), 
                BLUE, 2)

        if pos["x"] >= TOP_LEFT_X and pos["x"] <= int(utils.SCREEN_WIDTH/2)+TOP_LEFT_X and \
            pos["y"] >= TOP_LEFT_Y and pos["y"] <= int(utils.SCREEN_HEIGHT/2)+TOP_LEFT_Y:
            cv2.circle(image, (pos["x"], pos["y"]), 10, GREEN, cv2.FILLED, 2)
        
        else:
            cv2.circle(image, (pos["x"], pos["y"]), 10, RED, cv2.FILLED, 2)


    def run_mediapipe(self):
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    pos = self.get_landmark_pos(results.multi_face_landmarks, NOSE, image)
                    self.convert_landmark_pos_to_game_pos(pos)
                    self.draw_landmark(pos, image)

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()

f = FaceTracker()
f.run_mediapipe()