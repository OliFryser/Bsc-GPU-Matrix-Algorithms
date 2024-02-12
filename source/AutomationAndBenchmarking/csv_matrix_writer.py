import csv
import random

rows = 10000
columns = 10000

csv_file_path = '../BenchmarkingMatrixCSVs/'+str(rows)+'x'+str(columns)+'.csv'

try:
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([rows, columns])
        for _ in range(rows):
            random_numbers = [float("{:.1f}".format(random.uniform(0.0, 100.0))) for _ in range(columns)]  
            csv_writer.writerow(random_numbers)
except Exception as e:
    print(f"An error occurred: {e}")

