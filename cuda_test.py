import os

# Assuming CUDA_PATH is correctly set in your environment variables
cuda_path = os.environ["CUDA_PATH"]  # This should give you the base path
bin_path = os.path.join(cuda_path, "bin")  # Correctly joins the 'bin' directory

# Use this path to add to DLL directory
os.add_dll_directory(bin_path)
