import os
import numpy as np
import scipy.io as scio

# Path to the specific file you want to inspect
FILE_PATH = r"D:\last_works\code\data\PU_raw\K002\N09_M07_F10_K002_1.mat"


def inspect_mat_structure(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return

    print(f"Inspecting: {path}")
    mat = scio.loadmat(path)

    # Filter out system keys
    real_keys = [k for k in mat.keys() if not k.startswith('__')]

    if not real_keys:
        print("[WARN] No variables found in .mat file.")
        return

    # Usually there is only one top-level key matching the filename
    root_key = real_keys[0]
    print(f"Root Variable: '{root_key}'")

    root_data = mat[root_key]

    # Check for PU Dataset Structure (Struct containing 'Y')
    if root_data.dtype.names and 'Y' in root_data.dtype.names:
        print("Structure detected: Standard PU Dataset format")

        # Access the 'Y' struct (Measurement Data)
        # root_data is (1,1), so we index [0,0]
        y_struct = root_data[0, 0]['Y']

        # y_struct contains an array of sensor structs
        # We iterate through the first dimension (sensors)
        sensors = y_struct[0]

        print(f"\n{'Sensor Name':<25} | {'Shape':<15} | {'First Value'}")
        print("-" * 60)

        for i, sensor in enumerate(sensors):
            # Extract Name
            sensor_name = "Unknown"
            if 'Name' in sensor.dtype.names:
                # Name is usually stored as an array of strings
                try:
                    sensor_name = str(sensor['Name'][0])
                except:
                    pass

            # Extract Data Shape
            data_shape = "No Data"
            first_val = "N/A"

            if 'Data' in sensor.dtype.names:
                data = sensor['Data']
                data_shape = str(data.shape)
                if data.size > 0:
                    first_val = f"{data.flatten()[0]:.4f}"

            print(f"{sensor_name:<25} | {data_shape:<15} | {first_val}")

    else:
        print("[INFO] 'Y' struct not found. Dumping top-level fields:")
        print(root_data.dtype.names)


if __name__ == '__main__':
    inspect_mat_structure(FILE_PATH)