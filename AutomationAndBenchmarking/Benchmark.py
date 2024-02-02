import subprocess
import os

directory = "MatrixAlgorithms/"
program_name = "HelloWorld"

c_source_path = directory + program_name + ".c"
binary_path = directory + program_name
compile_command = ["gcc", c_source_path, "-o", binary_path]

try:
    subprocess.run(compile_command, check=True)
    subprocess.run([binary_path], check=True)
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