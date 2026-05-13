import dlib
import numpy as np
import argparse
import cv2
import sys
import os

EYES_INDICES  = list(range(36, 48, 1))
MOUTH_INDICES = list(range(49, 68, 1))

RIGHT_BROW_START_IND = 22
LEFT_BROW_START_IND  = 21
RIGHT_BROW_END_IND = 26
LEFT_BROW_END_IND  = 17

OFFSET = 0.1

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face leather cover extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--image", type=str, required=True,
                        help="path to image.")
    parser.add_argument("-p", "--predictor", type=str, default="shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib model shape_predictor.")
    parser.add_argument("-o", "--output", type=str, default="result.png",
                        help="Path for saving result.")
    return parser.parse_args()

def process_image(args: argparse.Namespace) -> np.ndarray:
    if not os.path.isfile(args.image):
        sys.exit(f"Image {args.image} not found")
    if not os.path.isfile(args.predictor):
        sys.exit(f"Model {args.predictor} not found")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Failed reading image {args.image}, wrong file.")

    faces = detector(img, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_mask = np.zeros_like(gray, dtype=np.uint8)

    for face in faces:
        shape = predictor(gray, face)

        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        eyes = landmarks[EYES_INDICES]
        mouth = landmarks[MOUTH_INDICES]

        left_eye = eyes[:6]
        right_eye = eyes[6:]

        eyes_mask = np.zeros_like(gray, dtype=np.uint8)
        mouth_mask = np.zeros_like(gray, dtype=np.uint8)

        face_hull = cv2.convexHull(landmarks)
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        mouth_hull = cv2.convexHull(mouth)

        cv2.fillConvexPoly(face_mask, face_hull, 255)
        cv2.fillConvexPoly(eyes_mask, left_eye_hull, 255)
        cv2.fillConvexPoly(eyes_mask, right_eye_hull, 255)
        cv2.fillConvexPoly(mouth_mask, mouth_hull, 255)
        face_mask -= eyes_mask
        face_mask -= mouth_mask

        left_brow_end  = landmarks[LEFT_BROW_END_IND]
        right_brow_end = landmarks[RIGHT_BROW_END_IND]
        top_y = face.top()
        
        face_width = face.right() - face.left()
        h_offset   = int(face_width * OFFSET)

        forehead_poly = np.array([
        [left_brow_end[0] - h_offset, top_y],
        [right_brow_end[0] + h_offset, top_y],
        right_brow_end,
        landmarks[RIGHT_BROW_START_IND],                    
        np.mean(landmarks[21:23], axis=0), # between brows
        landmarks[LEFT_BROW_START_IND],
        left_brow_end
        ], dtype=np.int32)

        cv2.fillPoly(face_mask, [forehead_poly], 255)

    face = cv2.bitwise_and(img, img, mask=face_mask)
    cv2.imwrite(args.output, face)


def main() -> None:
    args = parse_arguments()
    process_image(args)

if __name__ == "__main__":
    main()