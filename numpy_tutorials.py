#NumPy is a Python library.
#NumPy is used for working with arrays.
#NumPy is short for Numerical Python
#NumPy aims to provide an array object that is up to 50x faster than traditional Python lists.
#The array object in NumPy is called ndarray]

"""Why is NumPy Faster Than Lists?
NumPy arrays are stored at one continuous place in memory unlike lists, 
so processes can access and manipulate them very efficiently.
This behavior is called locality of reference in computer science."""

#Now NumPy is imported and ready to use
#1. Array Creation and Manipulation
import numpy as np
arr = np.array([1,2,3,4,5,6])
print(arr)

#nNumPy is usually imported under the np alias
#alias: In Python alias are an alternate name for referring to the same thing

import numpy  as np
arr = np.array([1,2,3,4,5,7,8,9,])
print(arr)

#Checking NumPy Version
import numpy as np 
print(np.__version__)

#0-D arrays(conatains only one element), or Scalars, are the elements in an array. 
#Each value in an array is a 0-D array.
import numpy as np
arr = np.array(42) #here 42 is a array value
print(arr)

"""An array that has 0-D arrays 
as its elements is called uni-dimensional or 1-D array."""
import numpy as np
arr = np.array([1,2,3,4,5,6,7,8,9])
print(arr)

#An array that has 1-D arrays as its elements is called a 2-D array
#NumPy has a whole sub module dedicated towards matrix operations called numpy.mat
import numpy as np
arr = np.array([[1,2,3,],[4,5,6]])
print (arr)

#An array that has 2-D arrays (matrices) as its elements is called 3-D array.
import numpy as np
arr = np.array([[[1,2,3] , [4,5,6]],[[1,2,3],[4,5,6]]])
print (arr)

#umPy Arrays provides the ndim attribute that returns an integer 
#that tells us how many dimensions the array have
import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

#When the array is created, you can define the number of dimensions by using the ndmin argument
import numpy as np
arr = np.array([1,2,3,4] , ndmin = 5)
print(arr)
#verify that it has 5 dimensions
print(arr)
print('number of dimensions : ' , arr.ndim)

#Get the first element from the following array
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[0])

#Access 2-D Arrays
#Access the element on the first row, second column
import numpy as np
arr = np.array([[1,2,3,4,0] , [6,7,8,9,10]])
print("2nd ele on 1st row :" , arr[0,1])
#Access the element on the 2nd row, 5th column
print("ele one 2nd row and 5th column" , arr[1,4])

#Access 2-D Arrays
#Access the third element of the second array of the first array
import numpy as np
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0, 1, 2])

#Print the last element from the 2nd dim:
import numpy as np
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('Last element from 2nd dim: ', arr[1, -1])

""""Slicing in python means taking elements from one given index to another given index.

We pass slice instead of index like this: [start:end].
We can also define the step, like this: [start:end:step].
If we don't pass start its considered 0
If we don't pass end its considered length of array in that dimension
If we don't pass step its considered 1"""

#slice elements from index 1 to 5 from following array
import numpy as np
arr = np.array([1,2,3,4,5,6])
print(arr[1:5])
#Note:The result includes the start index, but excludes the end index.

#Slice elements from index 4 to the end of the array:
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[4:])

#Slice elements from start to the index 4 of the array:
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[:4])

#Negative Slicing
#Slice from the index 3 from the end to index 1 from the end:
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[-3:-1])

#Use the step value to determine the step of the slicing:
import numpy as np
arr = np.array([1,2,3,4,5,6,7,8,9])
print(arr[1:5:2])

#Return every other element from the entire array:
import numpy as np
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[::2])

#Slicing 2-D Arrays
#From the second element, slice elements from index 1 to index 4 (not included):
import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4])

#From both elements, return index 2:
import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 2])
"returns ele at index 2 from both sliced array of both the array"

#From both elements, slice index 1 to index 4 (not included), this will return a 2-D array:
import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 1:4])


"""strings - used to represent text data, the text is given under quote marks. e.g. "ABCD"
integer - used to represent integer numbers. e.g. -1, -2, -3
float - used to represent real numbers. e.g. 1.2, 42.42
boolean - used to represent True or False.
complex - used to represent complex numbers. e.g. 1.0 + 2.0j, 1.5 + 2.5j

NumPy has some extra data types, and refer to data types with one character, like i for integers, u for unsigned integers etc.

Below is a list of all data types in NumPy and the characters used to represent them.

i - integer
b - boolean
u - unsigned integer
f - float
c - complex float
m - timedelta
M - datetime
O - object
S - string
U - unicode string
V - fixed chunk of memory for other type ( void )
"""
#Checking the Data Type of an Array(dtype)
import numpy as np
arr = np.array([1,2,3,4])
print(arr.dtype)

#Create an array with data type string:
import numpy as np
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
print(arr.dtype)

#Create an array with data type 4 bytes integer:
import numpy as np
arr = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(arr.dtype)

#Change data type from float to integer by using 'i' as parameter value:
import numpy as np
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print(newarr)
print(newarr.dtype)

#Get the Shape of an Array(shape)
"""NumPy arrays have an attribute called shape that returns a tuple with 
each index having the number of corresponding elements"""
import numpy as np
arr = np.array([[1,2,3,4], [5,6,7,8]])
print(arr.shape)
#above returns (2,4) indicates tht array has 2 dimension and eacha array contains 4 elements

"""Create an array with 5 dimensions using ndmin using a vector with values 1,2,3,4 and verify that last dimension has value 4:"""
import numpy as np
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('shape of array :', arr.shape)

"""Reshaping arrays
Reshaping means changing the shape of an array.
The shape of an array is the number of elements in each dimension.
By reshaping we can add or remove dimensions or change number of elements in each dimension."""

import numpy as np
arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr = arr.reshape(4,3)
print(newarr)

#creating arrays:array(): Converts lists or tuples into NumPy arrays.
import numpy as np
arr = np.array([1,2,3])
print(arr)

#zeros(): Creates an array filled with zeros.
zeros_array = np.zeros((2, 3))
print(zeros_array)

#ones():create an array filled with one's
one_arr = np.ones((2,3))
print(one_arr)

#empty()-creates an uninitialised array(random values)
empty_array = np.empty((2,3))
print(empty_array )

#arange(): Creates arrays with regularly spaced values.
arange_arr = np.arange(0,10,2)
print(arange_arr)

#linspace(): Creates arrays with evenly spaced values over a specified range.
linspace_arr = np.linspace(0, 1, 5)
print(linspace_arr)

#Array Properties:
#shape: Returns the dimensions of the array.
print(arr.shape)

#dtype: Data type of the array elements.
print(arr.dtype)

#size: Total number of elements in the array
print(arr.size)

#ndim: Number of dimensions of the array.
print(arr.ndim)

#Reshaping and Flattening Arrays:
#reshape(): Changes the shape of an array without changing its data.
reshaped_arr = arr.reshape(3, 1)
print(reshaped_arr)

#ravel(): Returns a flattened array (1D view).
ravel_arr = reshaped_arr.ravel()
print(ravel_arr)

#flatten(): Returns a copy of the array collapsed into one dimension.
flatten_arr = reshaped_arr.flatten()
print(flatten_arr)

#One-Dimensional and Multi-Dimensional Array Slicing:
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2[0, 1])  # Access element 2
print(arr2[:, 1])  # Access second column

#Boolean Indexing and Masking:
mask = arr2 > 3
print(arr2[mask])

# Array Operations
arr3 = np.array([1, 2, 3])
arr4 = np.array([4, 5, 6])
print(arr3 + arr4)
print(arr3 * arr4)

#Broadcasting Rules:
arr5 = np.array([[1], [2], [3]])
arr6 = np.array([4, 5, 6])
print(arr5 + arr6)

#Aggregation Functions:
print(arr3.sum())
print(arr3.mean())
print(arr3.std())

#Linear Algebra with NumPy
#Dot Product:
print(np.dot(arr3, arr4))
print(arr3 @ arr4)

#Matrix Multiplication:
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
print(np.matmul(mat1, mat2))

#transpose
print(np.matmul(mat1, mat2).T)

#Eigenvalues and Eigenvectors:
#An eigenvector of a square matrix A is a nonzero vector that only gets scaled (not rotated) when multiplied by the matrix.
#The scaling factor is called the eigenvalue (λ).
values, vectors = np.linalg.eig(mat1)
print(values)
print(vectors)

#Solving Linear Systems:
A = np.array([[3,1],[1,2]])
b = np.array([9,8])
print(np.linalg.solve(A,b))

#Inverse and Determinant:
print(np.linalg.inv(mat1))
print(np.linalg.det(mat1))

#Random Numbers and Statistics
#Generating Random Numbers:
print(np.random.rand(2, 3))
print(np.random.randn(2, 3))
print(np.random.randint(1, 10, (2, 3)))

#Seeding for Reproducibility:
np.random.seed(42)
print(np.random.rand(3))

#Sampling from Distributions:
print(np.random.choice([1, 2, 3, 4, 5], 3))
print(np.random.normal(0, 1, 3))

#basic statstical operations
print(np.median(arr3))
print(np.percentile(arr3, 50))

# Advanced Array Manipulation
print(np.vstack((arr3, arr4))) #vertical stack
print(np.hstack((arr3, arr4))) #horizontal stack

#Splitting Arrays
arr7 = np.array([1, 2, 3, 4, 5, 6])
print(np.split(arr7, 3))

#Tile and Repeat:
print(np.tile(arr3, 2))
print(np.repeat(arr3, 2))

#Sorting Arrays:
unsorted_arr = np.array([3, 1, 2])
print(np.sort(unsorted_arr))
print(np.argsort(unsorted_arr))
#np.argsort(unsorted_arr) would return [1, 2, 0], indicating that the 
#smallest element is at index 1, the next smallest is at index 2, and the largest is at index 0.

#Handling NaN:1, NaN (Not a Number), and 3
arr_with_nan = np.array([1, np.nan, 3])
print(np.isnan(arr_with_nan)) #The next line, print(np.isnan(arr_with_nan)), uses the np.isnan function to check each element of the array for NaN values
print(np.nan_to_num(arr_with_nan))#he final line, print(np.nan_to_num(arr_with_nan)), uses the np.nan_to_num function to replace NaN values in the array with zero

# Performance Optimization
arr_large = np.arange(10000)
print(arr_large * 2) #vectorized

#Broadcasting for Optimized Operations:
arr8 = np.array([1,2,3])
print(arr8 + 5)

#Memory-Efficient Slicing (No Copying):
view = arr_large[:5]
view[0] = 99
print(arr_large[:5]) # Reflects in original array

#Loading and Saving Arrays:
np.save('array.npy', arr3)
loaded_arr = np.load('array.npy')
print(loaded_arr)

np.savetxt('array.txt', arr3)
loaded_txt_arr = np.loadtxt('array.txt')
print(loaded_txt_arr)

#Unique Elements:
arr9 = np.array([1, 2, 2, 3, 4, 4])
print(np.unique(arr9))

#element-wise comoarison
arr10 = np.array([1, 2, 3, 4, 5])
print(np.where(arr10 > 3, 'Yes', 'No'))
print(np.logical_and(arr10 > 2, arr10 < 5))

#clipping
arr11 = np.array([1, 5, 10, 15])
print(np.clip(arr11, 3, 10))
#The function will limit the values in arr11 such that any value less than 3 is replaced by 3, and any value greater than 10 is replaced by 10.