from ultralytics import YOLO
import cv2
import time
import csv
from analytics import TrafficAnalytics

analytics = TrafficAnalytics() #create instance of analytics class


model = YOLO("yolov8n.pt")

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

csv_file = open('Traffic_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)                       #store data in csv file
csv_writer.writerow(["timestamp", "track_id", "class", "direction", "flow_rate"])   


car_count = 0
truck_count = 0                 #to count cars, trucks, buses individually
bus_count = 0

counted_ids = set() #container for the counted vehicle ids
vehicle_count = 0 #increases as cars pass he line
prev_positions = {} #to track previous positions of vehicles

start_time = time.time() #to calculate vehicle per minute

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

        elapsed_time = round(time.time() - start_time, 2)
        flow_rate = round(vehicle_count / (elapsed_time / 60) if elapsed_time > 0 else 0, 2)  #veh/min

        # Now only count vehicles that cross line from above to below
        if prev_y < line_y and center_y > line_y and track_id not in counted_ids:
            counted_ids.add(track_id)
            
            analytics.update_counts(cls)
            # vehicle_count += 1

            # if cls == 2:
            #     car_count += 1
            # elif cls == 7:
            #     truck_count += 1
            # elif cls == 5:
            #     bus_count += 1

            
            timestamp = round(time.time() - start_time, 2)
            csv_writer.writerow([timestamp, track_id, cls, "down", flow_rate])  #write data to csv
            
        elapsed_time = round(time.time() - start_time, 2)
        flow_rate = analytics.compute_flow_rate()

        prev_positions[track_id] = center_y



    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    cv2.putText(annotated_frame, f"Total Vehicle count: {vehicle_count}", (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Cars: {car_count}", (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)               #display individual counts of cars, buses, trucks on the frame

    cv2.putText(annotated_frame, f"Buses: {bus_count}", (50, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.putText(annotated_frame, f"Trucks: {truck_count}", (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.putText(annotated_frame, f"Flow Rate (vehicles/min): {flow_rate:.2f}", (50, 190),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)


    cv2.imshow("Traffic Tracking", annotated_frame)            #display 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
