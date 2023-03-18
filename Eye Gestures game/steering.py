import math
import keyinput
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX
# 0 For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    imageHeight, imageWidth, _ = image.shape

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    co=[]
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        for point in mp_hands.HandLandmark:
           if str(point) == "HandLandmark.WRIST":
              normalizedLandmark = hand_landmarks.landmark[point]
              pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                        normalizedLandmark.y,
                                                                                    imageWidth, imageHeight)

              try:
                co.append(list(pixelCoordinatesLandmark))
              except:
                  continue

    if len(co) == 2:
        xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
        radius = 150
        try:
            m=(co[1][1]-co[0][1])/(co[1][0]-co[0][0])
        except:
            continue
        a = 1 + m ** 2
        b = -2 * xm - 2 * co[0][0] * (m ** 2) + 2 * m * co[0][1] - 2 * m * ym
        c = xm ** 2 + (m ** 2) * (co[0][0] ** 2) + co[0][1] ** 2 + ym ** 2 - 2 * co[0][1] * ym - 2 * co[0][1] * co[0][
            0] * m + 2 * m * ym * co[0][0] - 22500
        xa = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        xb = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        ya = m * (xa - co[0][0]) + co[0][1]
        yb = m * (xb - co[0][0]) + co[0][1]
        if m!=0:
          ap = 1 + ((-1/m) ** 2)
          bp = -2 * xm - 2 * xm * ((-1/m) ** 2) + 2 * (-1/m) * ym - 2 * (-1/m) * ym
          cp = xm ** 2 + ((-1/m) ** 2) * (xm ** 2) + ym ** 2 + ym ** 2 - 2 * ym * ym - 2 * ym * xm * (-1/m) + 2 * (-1/m) * ym * xm - 22500
          try:
           xap = (-bp + (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
           xbp = (-bp - (bp ** 2 - 4 * ap * cp) ** 0.5) / (2 * ap)
           yap = (-1 / m) * (xap - xm) + ym
           ybp = (-1 / m) * (xbp - xm) + ym

          except:
              continue

        cv2.circle(img=image, center=(int(xm), int(ym)), radius=radius, color=(195, 255, 62), thickness=15)

        l = (int(math.sqrt((co[0][0] - co[1][0]) ** 2 * (co[0][1] - co[1][1]) ** 2)) - 150) // 2
        cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (195, 255, 62), 20)
        if co[0][0] > co[1][0] and co[0][1]>co[1][1] and co[0][1] - co[1][1] > 65:
            print("Turn left.")
            keyinput.release_key('s')
            keyinput.release_key('d')
            keyinput.press_key('a')
            cv2.putText(image, "Turn left", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(image, (int(xbp), int(ybp)), (int(xm), int(ym)), (195, 255, 62), 20)


        elif co[1][0] > co[0][0] and co[1][1]> co[0][1] and co[1][1] - co[0][1] > 65:
            print("Turn left.")
            keyinput.release_key('s')
            keyinput.release_key('d')
            keyinput.press_key('a')
            cv2.putText(image, "Turn left", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(image, (int(xbp), int(ybp)), (int(xm), int(ym)), (195, 255, 62), 20)


        elif co[0][0] > co[1][0] and co[1][1]> co[0][1] and co[1][1] - co[0][1] > 65:
            print("Turn right.")
            keyinput.release_key('s')
            keyinput.release_key('a')
            keyinput.press_key('d')
            cv2.putText(image, "Turn right", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(image, (int(xap), int(yap)), (int(xm), int(ym)), (195, 255, 62), 20)

        elif co[1][0] > co[0][0] and co[0][1]> co[1][1] and co[0][1] - co[1][1] > 65:
            print("Turn right.")
            keyinput.release_key('s')
            keyinput.release_key('a')
            keyinput.press_key('d')
            cv2.putText(image, "Turn right", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(image, (int(xap), int(yap)), (int(xm), int(ym)), (195, 255, 62), 20)
        
        else:
            print("keeping straight")
            keyinput.release_key('s')
            keyinput.release_key('a')
            keyinput.release_key('d')
            keyinput.press_key('w')
            cv2.putText(image, "keep straight", (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            if ybp>yap:
                cv2.line(image, (int(xbp), int(ybp)), (int(xm), int(ym)), (195, 255, 62), 20)
            else:
                cv2.line(image, (int(xap), int(yap)), (int(xm), int(ym)), (195, 255, 62), 20)

    if len(co)==1:
       print("keeping back")
       keyinput.release_key('a')
       keyinput.release_key('d')
       keyinput.release_key('w')
       keyinput.press_key('s')
       cv2.putText(image, "keeping back", (50, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

# Flip the image horizontally for a selfie-view display.
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
