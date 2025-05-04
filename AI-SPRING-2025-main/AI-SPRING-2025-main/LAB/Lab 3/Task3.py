import heapq
import random
import time

class DeliveryRouteOptimizer:
    def __init__(self, points, start, time_windows):
        self.points = points
        self.start = start
        self.time_windows = time_windows
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def heuristic(self, point, current_time):
        return min(abs(point[0] - p[0]) + abs(point[1] - p[1]) for p in self.points if current_time <= self.time_windows[p][1])

    def is_within_time_window(self, point, current_time):
        start, end = self.time_windows[point]
        return start <= current_time <= end

    def greedy_best_first_search(self):
        pq = []
        heapq.heappush(pq, (0, self.start, [self.start], 0))
        visited = set()
        deliveries = set(self.points)

        while pq and deliveries:
            _, current, path, current_time = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current in deliveries and self.is_within_time_window(current, current_time):
                deliveries.remove(current)
                if not deliveries:
                    return path

            for dx, dy in self.directions:
                new_point = (current[0] + dx, current[1] + dy)
                if new_point in self.points:
                    travel_time = 1
                    new_time = current_time + travel_time
                    if new_point not in visited and self.is_within_time_window(new_point, new_time):
                        heapq.heappush(pq, (self.heuristic(new_point, new_time), new_point, path + [new_point], new_time))
        
        return None

points = [(0, 5), (4, 5), (2, 2)]
start = (0, 0)
time_windows = {
    (0, 5): (2, 10),
    (4, 5): (5, 15),
    (2, 2): (1, 8)
}

optimizer = DeliveryRouteOptimizer(points, start, time_windows)
route = optimizer.greedy_best_first_search()

if route:
    print("Optimized Delivery Route:", route)
else:
    print("No feasible route found")
