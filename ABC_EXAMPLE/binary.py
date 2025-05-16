import numpy as np

input_file = 'inputcsv'
output_file = 'output.csv'

data = np.loadtxt(input_file, delimiter=',')

binarized = (data != 0).astype(int)

np.savetxt(output_file, binarized, fmt='%d', delimiter=',')

print(f"Binarized wm saved to '{output_file}'")
