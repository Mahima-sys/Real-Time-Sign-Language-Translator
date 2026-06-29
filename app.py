import cv2
import pyttsx3

from utils.detector import HandDetector
from utils.data_collector import DataCollector
from utils.predictor import Predictor


def main():
    # Initialize webcam
    camera = cv2.VideoCapture(0)

    detector = HandDetector()
    collector = DataCollector()
    predictor = Predictor()

    # Text-to-Speech
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    # Variables
    sentence = ""
    prediction = ""
    confidence = 0

    # Stable prediction
    last_prediction = ""
    stable_prediction = ""
    stable_count = 0
    STABLE_THRESHOLD = 10

    # Dataset letters
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    current_index = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Failed to access webcam.")
            break

        # Detect hand
        frame, landmarks = detector.find_hands(frame)

        prediction = ""
        confidence = 0

        if landmarks:

            prediction, confidence = predictor.predict(landmarks)

            # Stable prediction logic
            if prediction == last_prediction:
                stable_count += 1
            else:
                stable_count = 1
                last_prediction = prediction

            if stable_count >= STABLE_THRESHOLD:
                stable_prediction = prediction

        # Current dataset letter
        current_letter = letters[current_index]

        # Sample count
        sample_count = collector.get_sample_count(current_letter)

        # ---------------- Display ---------------- #

        cv2.putText(
            frame,
            f"Letter: {current_letter}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Stable: {stable_prediction}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Sentence: {sentence}",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "S=Save  N=Next  P=Prev  SPACE=Add  T=Speak  X=Clear  Q=Quit",
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Real-Time Sign Language Translator", frame)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Save dataset sample
        if landmarks and key == ord("s"):
            collector.save_sample(current_letter, landmarks)
            print(f"Saved {current_letter}")

        # Next letter
        elif key == ord("n"):
            if current_index < len(letters) - 1:
                current_index += 1

        # Previous letter
        elif key == ord("p"):
            if current_index > 0:
                current_index -= 1

        # Add stable prediction to sentence
        elif key == 32:  # Space
            if stable_prediction != "":
                sentence += stable_prediction

        # Speak sentence
        elif key == ord("t"):
            if sentence.strip() != "":
               engine = pyttsx3.init()
               engine.setProperty("rate", 150)
               engine.setProperty("volume", 1.0)

               speak_text = " ".join(sentence)
            print("Speaking:", speak_text)

            engine.say(speak_text)
            engine.runAndWait()
            engine.stop()

        # Delete last character
        elif key == 8:  # Backspace
            sentence = sentence[:-1]

        elif key == ord("x"):
           sentence = ""
           stable_prediction = ""
           last_prediction = ""
           stable_count = 0

        # Quit
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()