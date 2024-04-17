import math
import subprocess
import os
import csv_data_structure as structure
from data_visualizer import visualize_csv
from datetime import datetime

directory = "source/MatrixAlgorithms/"
directory_2d = "source/2DMatrixAlgorithms/"

cu_source_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".cu")]
cu_object_files = list(map(lambda file: file[:-len("cu")] + "o", cu_source_files))

c_source_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".c")]
c_object_files = list(map(lambda file: file[:-len("c")] + "o", c_source_files))

cu_source_files_2d = [os.path.join(directory_2d, name) for name in os.listdir(directory_2d) if name.endswith(".cu")]
cu_object_files_2d = list(map(lambda file: file[:-len("cu")] + "o", cu_source_files_2d))

c_source_files_2d = [os.path.join(directory_2d, name) for name in os.listdir(directory_2d) if name.endswith(".c")]
c_object_files_2d = list(map(lambda file: file[:-len("c")] + "o", c_source_files_2d))


binary_path = directory + "binary"
binary_path_2d = directory_2d + "binary"
compile_command = ["gcc", "-L/usr/local/cuda/lib64", "-o", binary_path] + c_object_files + cu_object_files + ["-lcunit", "-lcudart", "-lm"]
compile_command_2d = ["gcc", "-L/usr/local/cuda/lib64", "-o", binary_path_2d] + c_object_files_2d + cu_object_files_2d + ["-lcunit", "-lcudart"]

timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
csv_path = "BenchmarkData/" + timestamp + ".csv"
algorithms_to_run = ["diagnostic: launch kernel 1 block 1 thread", "diagnostic: launch kernel scaling grid and blocks", "diagnostic: cudaMalloc", "diagnostic: cudaMemcpy", "diagnostic: cudaMemcpy & launch kernel 1 block 1 thread", "diagnostic: cudaMemcpy & launch larger kernel"]
additional_csv_files_to_include = [] #["BenchmarkData/04-13 11:01:43 tampered.csv"] #["04-12 14:31:18 diagonstic 2.csv"] #["04-12 14:00:47 diagnostic1..csv"] #["BenchmarkData/03-01 11:47:22.csv"]
matrix_dimensions = [math.floor(2 ** (i+1)) for i in range(0, 12)] #, 1_000, 10_000, 100_000, 1_000_000]
diagram_save_path = "Diagrams/output_plot" + timestamp + ".png"

try:
    # Compile all c and cu files individually
    for (c_file, o_file) in zip(c_source_files, c_object_files):
        subprocess.run(["gcc", "-c", c_file, "-o", o_file])
    for (cu_file, o_file) in zip(cu_source_files, cu_object_files):
        subprocess.run(["nvcc", "-c", cu_file, "-o", o_file])

    #compile them all together
    subprocess.run(compile_command, check=True)

    #compile 2d
    for (c_file, o_file) in zip(c_source_files_2d, c_object_files_2d):
        subprocess.run(["gcc", "-c", c_file, "-o", o_file])
    for (cu_file, o_file) in zip(cu_source_files_2d, cu_object_files_2d):
        subprocess.run(["nvcc", "-c", cu_file, "-o", o_file])

    #compile them all together
    subprocess.run(compile_command_2d, check=True)
    
    for algorithm in algorithms_to_run:
        algorithm_is_2d = algorithm.startswith("2d")
        for dimension in matrix_dimensions:
            subprocess.run([binary_path_2d if algorithm_is_2d else binary_path, algorithm, str(dimension), csv_path], check=True)
    os.remove(binary_path)
    os.remove(binary_path_2d)
    for o_file in (c_object_files + cu_object_files + cu_object_files_2d + c_object_files_2d):
        os.remove(o_file)

except FileNotFoundError as e:
    print(f"File not found error: {e}")
    exit()
except subprocess.CalledProcessError as e:
    print(f"Error running the binary: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

if not os.path.exists(csv_path):
        with open(csv_path, 'w'):  # 'w' mode will create the file if it doesn't exist
            pass

data = structure.CSVDataStructure([csv_path] + additional_csv_files_to_include)
visualize_csv(data, diagram_save_path, algorithms_to_run)
