import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Function to draw structure lines
def draw_structure_lines(image, hand_landmarks):
    connections = mp_hands.HAND_CONNECTIONS
    if hand_landmarks is not None:
        for connection in connections:
            idx1, idx2 = connection
            x1 = int(hand_landmarks.landmark[idx1].x * image.shape[1])
            y1 = int(hand_landmarks.landmark[idx1].y * image.shape[0])
            x2 = int(hand_landmarks.landmark[idx2].x * image.shape[1])
            y2 = int(hand_landmarks.landmark[idx2].y * image.shape[0])
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check for like and dislike gestures
    like = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Assuming like gesture if the tip of the thumb is above the wrist
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                like = True

            # Draw the structure lines on the hand
            draw_structure_lines(frame, hand_landmarks)

    # Show like or dislike text based on gesture
    if like:
        cv2.putText(frame, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Hand Gestures', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
