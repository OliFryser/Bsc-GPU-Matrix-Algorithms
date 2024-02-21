import math
import subprocess
import os
import csv_data_structure as structure
from data_visualizer import visualize_csv
from datetime import datetime

directory = "source/MatrixAlgorithms/"
source_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".cu") or name.endswith(".c")]
binary_path = directory + "binary"
compile_command = ["nvcc", "-o", binary_path] + source_files
timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
data_append_file_csv = "BenchmarkData/" + timestamp + ".csv"
algorithms = ["addition cpu", "addition gpu single core", "addition gpu multi core"] # "multiplication", "inverse"]
matrix_dimensions = [math.floor(2 ** (i+1)) for i in range(0, 13)] #, 1_000, 10_000, 100_000, 1_000_000]
diagram_save_path = "Diagrams/output_plot" + timestamp + ".png"

try:
    subprocess.run(compile_command, check=True)
    for algorithm in algorithms:
        for dimension in matrix_dimensions:
            subprocess.run([binary_path, algorithm, str(dimension), data_append_file_csv], check=True)
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

data = structure.CSVDataStructure(data_append_file_csv)
visualize_csv(data, diagram_save_path, algorithms)