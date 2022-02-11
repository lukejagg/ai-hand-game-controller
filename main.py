# Mediapipe Import
import math
import threading
import time

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)
# mp_holistic = mp.solutions.holistic

# Keyboard / Mouse
import keyboard

import pyautogui as input

input.FAILSAFE = False
input.PAUSE = 0

pressed = {}


def press(str):
    if str not in pressed:
        if str == 'mouse_0':
            input.mouseDown()
        elif str == 'mouse_1':
            input.mouseDown(button='right')
        else:
            keyboard.press(str)
        pressed[str] = True


def release(str):
    if str in pressed:
        if str == 'mouse_0':
            input.mouseUp()
        elif str == 'mouse_1':
            input.mouseUp(button='right')
        else:
            keyboard.release(str)
        del pressed[str]


# Look Settings
look_speed = 0.16
yaw_speed = 300
roll_speed = 1.5 * 0
pitch_speed = 1.6
smooth = 0.5
hit_threshold = 0.005
smooth_zone = 0.85
place_threshold = 5
#place_threshold_forward = -140
place_head_threshold = -12

# Move Settings
backwards_threshold = -162
walk_threshold = -120 #-110
sprint_threshold = -70 #-80
sneak_threshold = 20
jump_threshold = -12
side_threshold_right = 35
side_threshold_left = -30


# Inventory Settings
hotbar_right_threshold = 15
hotbar_left_threshold = -25
hotbar_delay = 0.33
last_scroll = 0
open_inventory_threshold = 12
idle_threshold = -20

# Precise Move
#precise_threshold = -140
precise_threshold_slow = -135
#precise_sensitivity = 525
precise_sensitivity_slow = 400
using_precise = False
last_position = None

hand_delay = 0.25
left_start, right_start = 0, 0

# Variables
dx, dy = 0, 0
px, py = 0, 0
down, lastDown = False, False


def add(p1, p2):
    return (p1.x + p2.x, p1.y + p2.y, p1.z + p2.z)


def subtract(p1, p2):
    return (p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)


# Control
def mult(n):
    n2 = n * n
    return n2 * n / (n2 + 1)


def floor(n):
    if n >= 0:
        return math.floor(n)
    return math.ceil(n)


rx, ry = 0, 0


def move(x, y):
    d = math.sqrt(x * x + y * y)
    m = mult(d / smooth_zone) * smooth_zone

    global rx, ry
    dx = x * m + rx
    dy = y * m + ry

    fx = floor(dx)
    fy = floor(dy)

    rx = dx - fx
    ry = dy - fy

    input.moveRel(dx, dy)


center_roll = 0
center_pitch = 0
center_yaw = 0
center_roll, center_pitch, center_yaw = -0.36, 11.0, 0.00
#(0.48463738839214443, 10.935175195064275, -0.012546837329864502)
last_roll, last_pitch, last_yaw = 0, 0, 0


def ang(n):
    if n > 0:
        return -180 - (180 - n)
    return n


def analyze(hand, face, key):
    global using_precise
    global center_pitch, center_roll, center_yaw
    global left_start, right_start
    # Camera
    raw_roll, raw_pitch, raw_yaw = 0, 0, 0
    no_place = True
    if face and face.multi_face_landmarks:
        face_landmarks = face.multi_face_landmarks[0].landmark
        top = face_landmarks[151]
        bottom = face_landmarks[152]
        up_vector = subtract(top, bottom)

        middle = face_landmarks[0]
        right = subtract(middle, face_landmarks[261])
        left = subtract(middle, face_landmarks[31])

        raw_roll = math.degrees(math.atan2(up_vector[1], up_vector[0])) + 90
        raw_pitch = -math.degrees(math.atan2(up_vector[1], up_vector[2])) - 90
        raw_yaw = right[0] + left[0]

        # Calibration
        global last_roll, last_pitch, last_yaw
        if key == ord('c'):
            center_roll = raw_roll
            center_pitch = raw_pitch
            center_yaw = raw_yaw
            print((center_roll, center_pitch, center_yaw))

        roll = raw_roll - center_roll
        pitch = raw_pitch - center_pitch
        yaw = raw_yaw - center_yaw

        # roll = last_roll + (roll - last_roll) * smooth
        # pitch = last_pitch + (pitch - last_pitch) * smooth
        # yaw = last_yaw + (yaw - last_yaw) * smooth

        # last_roll, last_pitch, last_yaw = raw_roll, raw_pitch, raw_yaw

        # print((right, forward))
        size = input.size()
        sens = using_precise and 0.1 or 1
        move((yaw * yaw_speed * look_speed + roll * roll_speed * look_speed) * sens,
             (pitch * pitch_speed * look_speed) * sens)

        lips = subtract(face_landmarks[14], face_landmarks[13])
        if lips[1] > hit_threshold:
            press('mouse_0')
        else:
            release('mouse_0')

        if roll > open_inventory_threshold:
            press('e')
        else:
            release('e')

        if roll < place_head_threshold:
            press('mouse_1')
            no_place = False
        #else:
        #    no_place = True

    l = False
    r = False
    if hand and hand.multi_hand_landmarks and hand.multi_handedness:
        # print(', '.join("%s: %s" % item for item in vars(hand).items()))
        i = 0
        for i in range(0, len(hand.multi_handedness)):
            right = hand.multi_handedness[i].classification[0].label == "Right"
            hand_landmarks = hand.multi_hand_landmarks[i].landmark

            # Movement
            if not right:
                l = True
                if time.time() - left_start < hand_delay:
                    continue
                middle = hand_landmarks[9]
                top = hand_landmarks[12]
                vector = subtract(top, middle)
                deg = ang(math.degrees(math.atan2(vector[2], vector[1])))
                roll = math.degrees(math.atan2(vector[1], vector[0])) + 90

                # Forward / Back
                if deg < backwards_threshold:
                    press('s')
                else:
                    release('s')

                if deg > walk_threshold:
                    press('w')
                    if deg > sprint_threshold:
                        press('z')
                    else:
                        release('z')
                else:
                    release('w')
                    release('z')

                # Left / Right
                #if roll > side_threshold_right:
                #    press('d')
                #else:
                #    release('d')

                #if roll < -side_threshold_left:
                #    press('a')
                #else:
                #    release('a')

                # Crouching
                thumb_top = hand_landmarks[4]
                thumb_bottom = hand_landmarks[1]
                thumb_vector = subtract(thumb_top, thumb_bottom)
                thumb_deg = math.degrees(math.atan2(thumb_vector[1], thumb_vector[0])) + 90
                if thumb_deg > sneak_threshold:
                    press('shift')
                else:
                    release('shift')

                if thumb_deg < jump_threshold:
                    press(' ')
                else:
                    release(' ')
            else:
                r = True
                if time.time() - right_start < hand_delay:
                    continue
                # Right hand
                middle = hand_landmarks[9]
                top = hand_landmarks[12]
                vector = subtract(top, middle)
                deg = ang(math.degrees(math.atan2(vector[2], vector[1])))
                roll = math.degrees(math.atan2(vector[1], vector[0])) + 90

                global last_scroll
                t = time.time() - last_scroll
                if t > hotbar_delay:
                    last_scroll = time.time()
                    if roll > hotbar_right_threshold:
                        input.scroll(-1)
                    if roll < hotbar_left_threshold:
                        input.scroll(1)

                #if deg > place_threshold:
                #    press('mouse_1')
                #else:
                #    release('mouse_1')

                global last_position
                if deg < precise_threshold_slow: #deg > precise_threshold or
                    using_precise = True
                    pos = add(hand_landmarks[0], hand_landmarks[13])
                    sens = deg < precise_threshold_slow and precise_sensitivity_slow #or precise_sensitivity
                    if last_position is None:
                        last_position = pos
                    move((pos[0] - last_position[0]) * sens,
                         (pos[1] - last_position[1]) * sens)
                    last_position = pos
                else:
                    using_precise = False
                    last_position = None

                middle = subtract(hand_landmarks[12], hand_landmarks[11])
                pinkie = subtract(hand_landmarks[20], hand_landmarks[19])
                diff = pinkie[1] - middle[1]
                if diff > 0.04:
                    press('f')
                else:
                    release('f')

                thumb_top = hand_landmarks[4]
                thumb_bottom = hand_landmarks[1]
                thumb_vector = subtract(thumb_top, thumb_bottom)
                thumb_deg = math.degrees(math.atan2(thumb_vector[1], thumb_vector[0])) + 90
                #print(thumb_deg)
                if thumb_deg > place_threshold: #or deg > place_threshold_forward:
                    press('mouse_1')
                    no_place = False
                #else:
                #    no_place = True

    if not l:
        left_start = time.time()
        release(' ')
        release('shift')
        release('a')
        release('d')
        release('w')
        release('z')
        release('s')
    if not r:
        right_start = time.time()
        release('f')
        #release('mouse_1')
        using_precise = False
        last_position = None
    if no_place:
        release('mouse_1')


p_hand, p_face, p_key = None, None, None


def pseudo_analyze(hand, face, key):
    global p_hand, p_face, p_key
    p_hand, p_face, p_key = hand, face, key


# Debug
n1 = 0
n2 = 1
test_connections = frozenset({(n1, n2)})


def connect():
    global test_connections
    test_connections = frozenset({(n1, n2)})


# Multi-thread smoother
run = True


def thread_function(name):
    global p_hand, p_face, p_key
    time.sleep(1)
    c = 0
    t0 = time.time()
    while run:
        t = time.time() - t0
        f = 1.0 / 144
        if t > f:
            t = t - f
            analyze(p_hand, p_face, p_key)
            p_key = None
        time.sleep(1.0 / 2000.0)


thread = threading.Thread(target=thread_function, args=(1,))
thread.start()

# Webcam Loop
cap = cv2.VideoCapture(0)


def get_image():
    success, image = cap.read()
    if not success:
        return False

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return True, image


def draw_landmarks(image, hand_results, face_results):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Hand drawing
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Face drawing
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=test_connections,
                                      # mp_face_mesh.FACE_CONNECTIONS,
                                      landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

    # cv2.line(image, (0, 0), (10, 10), (255,255,255), thickness=1, lineType=4)
    cv2.imshow('Control', image)


# with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.25, min_tracking_confidence=0.25) as hands:
with mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            # Get image
            success, image = get_image()
            if not success:
                continue

            # Find key points
            hand_results = hands.process(image)
            face_results = face_mesh.process(image)
            # print(', '.join("%s: %s" % item for item in vars(face_results).items()))

            # Draw points
            draw_landmarks(image, hand_results, face_results)

            # Process
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord('p'):
                n2 = n2 + 1
                print(n2)
            if key == ord('o'):
                n1 = n1 + 1
                print(n1)
            connect()

            # Analyze
            pseudo_analyze(hand_results, face_results, key)

cap.release()
cv2.destroyAllWindows()
run = False
