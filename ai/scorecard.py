import csv
from io import StringIO

class Scorecard:
    def __init__(self):
        self.rows = []

    def add(self, q_no, question, answer, score=None, feedback=None, improvement_suggestions=None, confidence=None):
        self.rows.append({
            'q_no': q_no,
            'question': question,
            'answer': answer,
            'score': score,
            'feedback': feedback,
            'improvement_suggestions': improvement_suggestions,
            'confidence': confidence,
        })

    def to_csv(self):
        if not self.rows:
            return ""
        out = StringIO()
        writer = csv.DictWriter(out, fieldnames=self.rows[0].keys())
        writer.writeheader()
        for r in self.rows:
            writer.writerow(r)
        return out.getvalue()

    def save(self, path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if self.rows:
                writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
                writer.writeheader()
                for r in self.rows:
                    writer.writerow(r)
