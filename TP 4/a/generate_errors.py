import subprocess
import sys

# Parameters.
file_stem = sys.argv[1]
k_list = [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100]
n_iterations = 9
col_names = "k,train_err,valid_err,test_err\n"

err_file = open(f"k_nn_{file_stem}.err", "w")
err_file.write(col_names)
err_file.close()

# Generate errors.
for k in k_list:
    # Write k to config file.
    config_file = open(f"{file_stem}.knn", "r")
    lines = config_file.readlines()
    lines[0] = f"{k}\n"
    config_file = open(f"{file_stem}.knn", "w")
    config_file.writelines(lines)
    config_file.close()

    print(f"k: {k}")
    for iter in range(n_iterations):
        # Generate K-NN error.
        completed = subprocess.run(["python", "k_nn.py", file_stem])
        print(f"{iter + 1}/{n_iterations}")
