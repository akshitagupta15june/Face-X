import cv2


def draw_tracker(frame, tracker):
    for person in tracker.objects.values():
        text = f"ID: {person.id_}"
        cx, cy = person.centroid.astype("int")
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(
            frame, text, (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return frame


def draw_bounding_boxes(frame, bboxes, scores):
    for score, (x1, y1, x2, y2) in zip(scores, bboxes):
        text = f"({score:0.2%})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return frame
