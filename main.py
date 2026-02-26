from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

counted_ids = set() #container for the counted vehicle ids
vehicle_count = 0 #increases as cars pass he line

prev_positions = {} #to track previous positions of vehicles

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break                                   #read vid
    height, width, _ = frame.shape
    line_y = height // 2 #for hori line


    results = model.track(frame, persist=True)         #track

    boxes = results[0].boxes     #get boxes

    for box in boxes:
        if box.id is None:                      #if no id, skip
            continue

        track_id = int(box.id.item())
        cls = int(box.cls.item())

        if cls not in [2, 5, 7]:                                    # Only count vehicles (car=2, truck=7, bus=5 in COCO)
            continue

        x1, y1, x2, y2 = box.xyxy[0]                            
        center_y = int((y1 + y2) / 2)                               #centre of box

        if track_id not in prev_positions:
            prev_positions[track_id] = center_y
  
        prev_y = prev_positions[track_id]


        # Now only count vehicles that cross line from above to below
        if prev_y < line_y and center_y > line_y and track_id not in counted_ids:
            counted_ids.add(track_id)
            vehicle_count += 1
        prev_positions[track_id] = center_y

    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)  #draw hori line

    cv2.putText(frame, f"Vehicle Count: {vehicle_count}",       
            (50, 50),           
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)


    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    cv2.putText(annotated_frame, f"Vehicle Count: {vehicle_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

    cv2.imshow("Traffic Tracking", annotated_frame)

               
    cv2.imshow("Traffic Detection", annotated_frame)         #display         

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
