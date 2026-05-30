import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Initialize voice engine
engine = pyttsx3.init()

# Voice settings
engine.setProperty('rate', 150)

# Store previous coffee selection
previous_coffee = ""

# Finger tip landmark IDs
tip_ids = [4, 8, 12, 16, 20]

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    # Read webcam frame
    success, frame = cap.read()

    # Check if frame captured properly
    if not success:
        print("Failed to capture frame")
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    result = hands.process(rgb_frame)

    # Finger counter
    finger_count = 0

    # Coffee variable
    coffee = ""

    # If hand detected
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            landmarks = []

            # Get landmark positions
            for id, lm in enumerate(hand_landmarks.landmark):

                h, w, c = frame.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                landmarks.append((cx, cy))

            # Thumb detection
            if landmarks[tip_ids[0]][0] > landmarks[tip_ids[0] - 1][0]:
                finger_count += 1

            # Other fingers detection
            for id in range(1, 5):

                if landmarks[tip_ids[id]][1] < landmarks[tip_ids[id] - 2][1]:
                    finger_count += 1

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Coffee selection logic
    if finger_count == 1:
        coffee = "Espresso"

    elif finger_count == 2:
        coffee = "Latte"

    elif finger_count == 3:
        coffee = "Cappuccino"

    elif finger_count == 4:
        coffee = "Americano"

    elif finger_count == 5:
        coffee = "Mocha"

    else:
        coffee = "No Selection"

    # Voice assistant
    if coffee != previous_coffee and coffee != "No Selection":

        speech = f"Preparing {coffee}"

        print(speech)

        engine.say(speech)

        engine.runAndWait()

        previous_coffee = coffee

    # Display finger count
    cv2.putText(
        frame,
        f'Fingers: {finger_count}',
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display selected coffee
    cv2.putText(
        frame,
        f'Coffee: {coffee}',
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Display instructions
    cv2.putText(
        frame,
        "Show fingers to select coffee",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Show webcam window
    cv2.imshow("AI Coffee Machine", frame)

    # ESC key to exit
    if cv2.waitKey(1) == 27:
        break

# Release webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
