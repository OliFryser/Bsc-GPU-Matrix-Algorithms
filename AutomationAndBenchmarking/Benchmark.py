import subprocess
import os

directory = "MatrixAlgorithms/"
source_files = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".c")]
binary_path = directory + "binary"
compile_command = ["gcc", "-o", binary_path] + source_files

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