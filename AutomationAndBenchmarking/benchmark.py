import subprocess
import os
import csv
import matplotlib.pyplot as pyplot
import data_visualizer as visualizer
import csv_data_structure as structure
from data_visualizer import visualize_csv

directory = "MatrixAlgorithms/"
source_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".c")]
binary_path = directory + "binary"
compile_command = ["gcc", "-o", binary_path] + source_files
data_save_file = "DummyData/RandomData2.csv"
input_sizes = [5, 50, 500, 5_000, 50_000, 500_000, 5_000_000, 50_000_000]

try:
    subprocess.run(compile_command, check=True)
    for input_size in input_sizes:
        subprocess.run([binary_path, data_save_file, str(input_size)], check=True)
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


data = structure.CSVDataStructure(data_save_file)
visualize_csv(data)