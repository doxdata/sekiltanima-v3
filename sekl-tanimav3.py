import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


def is_parallel(shape):
    def angle_cos(p0, p1, p2):
        d1, d2 = (p0 - p1).astype(np.float32), (p2 - p1).astype(np.float32)
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    if len(shape) == 4:
        cos = [angle_cos(shape[i], shape[(i + 1) % 4], shape[(i + 2) % 4]) for i in range(4)]
        return np.all(np.array(cos) < 0.1)
    return False


def detect_shapes(frame):
    height, width = frame.shape[:2]
    start_x = width // 4
    start_y = height // 4
    end_x = 3 * width // 4
    end_y = 3 * height // 4

    roi = frame[start_y:end_y, start_x:end_x]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        shape_name = "Bilinmeyen"

        if area > 1000:
            if len(approx) == 3:
                shape_name = "Ucgen"
            elif len(approx) == 4:
                if is_parallel(approx.reshape(4, 2)):
                    shape_name = "Paralelkenar"
                else:
                    shape_name = "Dortgen"
            elif len(approx) == 5:
                shape_name = "Besgen"
            elif len(approx) == 6:
                shape_name = "Altigen"
            elif len(approx) >= 8:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                if 0.9 <= area / (np.pi * radius ** 2) <= 1.1:
                    shape_name = "Daire"
                    cv2.circle(roi, center, radius, (0, 255, 0), 2)
                    cv2.putText(roi, shape_name, (int(x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
                    continue

            if len(approx) >= 10 and area > 2000:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area
                if solidity > 0.5:
                    shape_name = "Yildiz"

            # Elips tespiti
            if len(contour) >= 5:  # Elips tespiti icin en az 5 nokta gerekli
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                if 0.8 <= axes[0] / 300 <= 1.2 and 0.8 <= axes[1] / 400 <= 1.2:
                    shape_name = "Elips"
                    cv2.ellipse(roi, ellipse, (0, 255, 0), 2)
                    cv2.putText(roi, shape_name, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
                    continue

            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(roi, shape_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    frame[start_y:end_y, start_x:end_x] = roi
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    return frame


def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = detect_shapes(frame)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, update_frame)


def release_camera():
    if cap.isOpened():
        cap.release()


def main():
    global cap, label
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Kamera acilamadi!")
        exit()

    root = tk.Tk()
    root.title("Sekil Algilama")
    root.protocol("WM_DELETE_WINDOW", release_camera)
    root.geometry("640x480")

    label = tk.Label(root)
    label.pack()

    update_frame()

    root.mainloop()

    release_camera()


if __name__ == "__main__":
    main()
