import cv2
import numpy as np

# Åbn webcam
cap = cv2.VideoCapture(0)

# Definer HSV-farveområder (skal justeres efter lysforhold)
lower_white = np.array([0, 0, 230])  # Very low saturation, very high brightness
upper_white = np.array([180, 20, 255])  # Only allows pure whites

lower_orange = np.array([5, 100, 100])
upper_orange = np.array([20, 255, 255])

lower_obstacle = np.array([10, 150, 150])
upper_obstacle = np.array([30, 255, 255])

def find_objects(mask, color_name, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # Justér efter størrelse af bordtennisbolde
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2  # Beregn centerkoordinater
            positions.append((cx, cy))  # Gem positionen
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, color_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return positions

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masker for hhv. hvide bolde, orange bold og forhindringer
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    obstacle_mask = cv2.inRange(hsv, lower_obstacle, upper_obstacle)

    # Find objekternes positioner
    white_positions = find_objects(white_mask, "White", frame)
    orange_position = find_objects(orange_mask, "Orange", frame)
    obstacle_positions = find_objects(obstacle_mask, "Obstacle", frame)

    # Vis resultaterne
    cv2.imshow("Processed Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
