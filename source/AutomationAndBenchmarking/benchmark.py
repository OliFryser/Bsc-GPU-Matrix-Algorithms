import math
import subprocess
import os
import csv_data_structure as structure
from data_visualizer import visualize_csv
from datetime import datetime

# 2d Algorithms
addition_cpu_2d = "2d addition cpu"
addition_gpu_single_core_2d = "2d addition gpu single core"
addition_gpu_multi_core_2d = "2d addition gpu multi core"

# Addition Algorithms
addition_cpu = "addition cpu"
addition_gpu_single_core = "addition gpu single core"
addition_gpu_multi_core = "addition gpu multi core"
addition_gpu_multi_core_2 = "addition gpu multi core 2"
addition_gpu_blocks = "addition gpu blocks"

# Multiplication Algorithms
multiplication_cpu = "multiplication cpu"
multiplication_gpu_single_core = "multiplication gpu single core"
multiplication_gpu_multi_core_unwrapping_i = "multiplication gpu multi core unwrapping i"
multiplication_gpu_multi_core_unwrapping_i_and_j = "multiplication gpu multi core unwrapping i and j"
shared_memory_multiplication = "shared memory multiplication"
shared_memory_fewer_accesses = "shared memory fewer accesses"

# QR Algorithms
qr_cpu = "qr cpu"
qr_gpu_single_core = "qr gpu single core"
qr_gpu_parallel_max = "qr gpu parallel max"
qr_gpu_multi_core_single_kernel = "qr gpu multi core single kernel"
qr_algorithms = [qr_cpu, qr_gpu_single_core, qr_gpu_parallel_max, qr_gpu_multi_core_single_kernel]

diagnostic_no_op = "diagnostic: launch kernel 1 block 1 thread"
diagnostic_scaling_grid_and_blocks = "diagnostic: single kernel with grid and block size = x"
diagnostic_malloc = "diagnostic: cudaMalloc x floats"
diagnostic_malloc_and_memcopy = "diagnostic: cudaMalloc & cudaMemcpy x floats"
diagnostic_malloc_copy_and_launch_kernel = "diagnostic: cudaMemcpy & launch kernel 1 block 1 thread"
diagnostic_malloc_copy_and_launch_kernel_larger = "diagnostic: cudaMemcpy & launch larger kernel"
diagnostic_launch_x_kernels = "diagnostic: launch x kernels"
diagnostic_launch_x_kernels_sequentially = "diagnostic: launch x kernels sequentially"
gpu_diagnostics = [diagnostic_scaling_grid_and_blocks, diagnostic_malloc, 
                   diagnostic_malloc_and_memcopy, diagnostic_launch_x_kernels,
                   diagnostic_launch_x_kernels_sequentially]

diagnostic_write_managed = "diagnostic: write managed"
diagnostic_write_vector = "diagnostic: write vector"
diagnostic_write = [diagnostic_write_managed, diagnostic_write_vector]

parallel_max = "parallel max"
sequential_max = "sequential max"
max = [parallel_max] + [sequential_max]

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

# TWEAK THESE
algorithms_to_run = ["M1: " + qr_gpu_parallel_max] #[addition_cpu, addition_cpu_2d, addition_gpu_single_core, addition_gpu_single_core_2d, addition_gpu_multi_core, addition_gpu_multi_core_2d] # ["diagnostic: launch kernel 1 block 1 thread", "diagnostic: launch kernel scaling grid and blocks", "diagnostic: cudaMalloc", "diagnostic: cudaMemcpy", "diagnostic: cudaMemcpy & launch kernel 1 block 1 thread", "diagnostic: cudaMemcpy & launch larger kernel"]
additional_algorithms_to_compare = [] #[qr_gpu_single_core, qr_gpu_parallel_max]
additional_csv_files_to_include = []

matrix_dimensions = [math.floor(2 ** (i)) for i in range(1, 11)] #, 1_000, 10_000, 100_000, 1_000_000]
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

data = structure.CSVDataStructure(([csv_path] if os.path.exists(csv_path) else []) + additional_csv_files_to_include)
visualize_csv(data, diagram_save_path, algorithms_to_run + additional_algorithms_to_compare)
