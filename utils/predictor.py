import joblib


class Predictor:

    def __init__(self):
        self.model = joblib.load("models/hand_sign_model.pkl")

    def predict(self, landmarks):

        row = []

        base_x = landmarks[0][1]
        base_y = landmarks[0][2]

        for landmark in landmarks:
            row.extend([
                landmark[1] - base_x,
                landmark[2] - base_y,
                landmark[3]
            ])

        prediction = self.model.predict([row])[0]

        confidence = max(self.model.predict_proba([row])[0]) * 100

        return prediction, confidence