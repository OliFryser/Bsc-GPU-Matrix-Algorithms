import subprocess
import os
import csv_data_structure as structure
from data_visualizer import visualize_csv

directory = "source/MatrixAlgorithms/"
source_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".c")]
binary_path = directory + "binary"
compile_command = ["gcc", "-o", binary_path] + source_files
data_save_file_csv = "DummyData/RandomData2.csv"
algorithms = ["addition", "multiplication", "inverse"]
matrix_dimensions = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
diagram_save_path = "Diagrams/output_plot.png"

try:
    subprocess.run(compile_command, check=True)
    for algorithm in algorithms:
        for dimension in matrix_dimensions:
            subprocess.run([binary_path, algorithm, str(dimension), data_save_file_csv], check=True)
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

data = structure.CSVDataStructure(data_save_file_csv)
visualize_csv(data, diagram_save_path)