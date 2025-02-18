class DynamicMovingAverage:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.vals = []

    def append(self, value):
        self.count += 1
        self.sum += value
        self.vals.append(value)
        return self.get_average()

    def get_average(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def get_sum(self):
        return self.sum

    def get_vals_list(self):
        return self.vals

    def reset(self):
        self.count = 0
        self.sum = 0
        self.vals.clear()

#
# # Example usage
# average_calculator = DynamicAverageCalculator()
#
# # Adding data points dynamically
# data_points = [10, 20, 30, 40, 50]
# for point in data_points:
#     current_average = average_calculator.add_data_point(point)
#     print(f"Added {point}, current average: {current_average:.2f}")
