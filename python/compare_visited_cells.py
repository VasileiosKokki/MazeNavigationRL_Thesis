import os

from python.utils import get_eval_path

# Compares the visited cells of experiment2 and live_experiment line by line, prints any differences, and confirms if they are identical.
# If they are identical, then the interface between the js game and gymnasium works correctly

def compare_files(file1, file2):
    # Check existence
    if not os.path.exists(file1):
        print(f"❌ File not found: {file1}")
        print(f"Run agent.py with --folder experiment2 and --eval flag and try_sb3 uncommented")
        return
    if not os.path.exists(file2):
        print(f"❌ File not found: {file2}")
        print(f"Run server.js with --eval flag")
        return

    # Read lines
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = [line.strip() for line in f1.readlines()]
        lines2 = [line.strip() for line in f2.readlines()]

    # Pad missing lines so lengths match
    max_len = max(len(lines1), len(lines2))
    lines1 += ["<MISSING>"] * (max_len - len(lines1))
    lines2 += ["<MISSING>"] * (max_len - len(lines2))

    # Compare
    differences_found = False
    for i, (l1, l2) in enumerate(zip(lines1, lines2), start=1):
        if l1 != l2:
            differences_found = True
            print(f"Line {i} differs:")
            print(f"  {file1}: {l1}")
            print(f"  {file2}: {l2}")

    if not differences_found:
        print("✅ Files are identical!")

# Example usage
eval_dir_1 = os.path.join("evaluations", "experiment2")
eval_dir_2 = os.path.join("evaluations", "live_experiment")
eval_path_1 = get_eval_path(eval_dir_1)
eval_path_2 = get_eval_path(eval_dir_2)

compare_files(eval_path_1, eval_path_2)