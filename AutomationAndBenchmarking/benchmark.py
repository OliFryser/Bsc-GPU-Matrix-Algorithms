import subprocess
import os
import csv
import matplotlib.pyplot as pyplot
import data_visualizer as visualizer

directory = "MatrixAlgorithms/"
source_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".c")]
binary_path = directory + "binary"
compile_command = ["gcc", "-o", binary_path] + source_files
data_save_file = "DummyData/RandomData.csv"
number_count = 5

# Run C-program to generate data
try:
    subprocess.run(compile_command, check=True)
    subprocess.run([binary_path, data_save_file, str(number_count)], check=True)
    os.remove(binary_path)
except FileNotFoundError as e:
    print(f"File not found error: {e}")
    exit()
except subprocess.CalledProcessError as e:
    print(f"Error running the binary: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# # Visualize data with matplotlib
# with open(data_save_file, 'r') as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         numbers = list(map(int, row))
#         x = list(range(number_count))
#         pyplot.plot(x, numbers)

x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [3, 2, 1]
y2 = [1.1, 2.1, 3.1]
pyplot.plot(x1, y1)
pyplot.plot(x2, y2)

pyplot.xticks(list(range(number_count)))
pyplot.ylabel("Number Generated")
pyplot.title(str(number_count) + " Random Numbers")
pyplot.savefig("Diagrams/output_plot.png")

visualizer.visualize_csv("some/path")