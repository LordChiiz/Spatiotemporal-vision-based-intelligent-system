import time

class TrafficAnalytics:
    def __init__(self):
        self.vehicle_count = 0
        self.car_count = 0
        self.bus_count = 0
        self.truck_count = 0
        self.start_time = time.time()
        self.crossing_times = []


    def update_counts(self, cls):
        self.vehicle_count += 1

        if cls == 2:
            self.car_count += 1
        elif cls == 5:
            self.bus_count += 1
        elif cls == 7:
            self.truck_count += 1
        current_time = time.time()
        self.crossing_times.append(current_time - self.start_time)

        self.crossing_times = [t for t in self.crossing_times if current_time - t <= 60]  # Keep only crossings in the last minute

    def compute_flow_rate(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time <= 0:
            return 0
        return len(self.crossing_times)
    
    