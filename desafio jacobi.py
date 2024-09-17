import numpy as np

# Define matrices A and B as seen in the image
A = np.array([[0, -2, -6],
              [1, 0, -2],
              [8, 5, 0]])

B = np.array([[-8],
              [-2],
              [13]])

# Inverse of matrix A
A_inv = np.linalg.inv(A)

# Matrix multiplication to get the result of A_inv * B
result = np.dot(A_inv, B)

# Alpha values from the image
alpha1 = 8
alpha2 = 2
alpha3 = 13

max_alpha = max(alpha1, alpha2, alpha3)

# Defining X values based on the image for different iterations
X = np.array([[0, -8, -74, 52, 0],
              [0, -2, -26, 0, 0],
              [0, 13, 0, 0, 0]])

# Calculate the error as difference between iterations
error1 = np.abs(X[0, :-1] - X[0, 1:])
error2 = np.abs(X[1, :-1] - X[1, 1:])
error3 = np.abs(X[2, :-1] - X[2, 1:])

# Maximum error across iterations
max_error = np.maximum.reduce([error1, error2, error3])

# Output results
print("Inverse of A:")
print(A_inv)
print("\nResult of A_inv * B:")
print(result)
print("\nMax Alpha:", max_alpha)
print("\nError 1:", error1)
print("Error 2:", error2)
print("Error 3:", error3)
print("\nMax Error in each iteration:", max_error)
