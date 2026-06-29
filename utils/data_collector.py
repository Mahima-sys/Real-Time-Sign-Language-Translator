import csv
import os


class DataCollector:

    def __init__(self):
        os.makedirs("dataset", exist_ok=True)

    def save_sample(self, label, landmarks):

        filename = f"dataset/{label}.csv"

        row = []

        # Wrist coordinates (Landmark 0)
        base_x = landmarks[0][1]
        base_y = landmarks[0][2]

        for landmark in landmarks:
            row.extend([
                landmark[1] - base_x,
                landmark[2] - base_y,
                landmark[3]
         ]) 

        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def get_sample_count(self, label):

        filename = f"dataset/{label}.csv"

        if not os.path.exists(filename):
            return 0

        with open(filename, "r") as file:
            return sum(1 for _ in file)