# NDArray
This is a small header only library aimed to provide a c++ n-dimensional tensor with access
and splice methods as close as possible to python numpy arrays, while providing compatibility with the STL through
random access iterators.

# Requirements
a c++20 compiler, like GCC 10, is required. The library uses concepts to improve readability.
 A backport to c++17 might be available in the future.
 
# Installation
The library is headers only, and only the folder `ndarray` needs to be copied to the include path.
Alternatively `cmake .; make; make install` will do the trick. Tests are built with cmake and
Google's gtest and benchmark libraries. 
 
# Examples
Arrays are created by passing the contained type and the number of dimensions.
```
nd::NDArray<int, 4> arr(5, 5, 5, 5); // Creates a 4D dimensional array of integer with 5^4 elements. 
```  

values can be accessed either by index,  `[]` for the linear index, `()` for Cartesian index, 
or through iterators. Assignments from scalars are broadcasted to the entire array.

```
arr = 1; // All elements are 1.
std::iota(arr.begin(), arr.end(), 0); // A continuos range of values 0, 1, 2... is assigned to 'arr'.
arr(1,2,3,4) = 42; // Assignment to single location.
```

## Slicing and views
Like with python, providing a range of indices, or fixing them across some dimensions, provide a view
to the original data of dimension equal or lesser. Unfortunately there is no `:` operator in c++,
so a range will have to be provided as `nd::range{start, end}` representing the semi open interval
`[start, end)`. The token `nd::end` is used to mark the end of the original range, and `nd::all`
is short hand for the entire range. Selecting a negative index or range end means counting from the end,
like in python. E.g `nd::range{0, -1}` takes all elements but the last one.

```
nd::NDArray<float, 3> tensor(3,3,3); // 3D tensor
auto m = tensor(-1, nd::all, nd::all); // 3x3 matrix with the first index of 'tensor' fixed to 2. 
m.shape(); // Array with elements {3, 3}
``` 

STL algorithms can be used both on arrays and views.

```
auto column = m(nd::all, 0); // first column
std::sort(column.begin(), column.end());
```

Views and arrays of the same shape can be assigned to each other. 
```
m(0, nd::all) = column; // assign column to row.
```


## Broadcasting
Lazy evaluations of arithmetic operators `+, -, *, /` is supported between tensors
of the same shape or scalars. The result is evaluated when assigned to an array or view. 
The measured performance of a compound lazy evaluations is within 10% of a manually written loop. 

```
// Initialize A, B, C tensors of dimension >= 3 and same shape
C(0, nd::all, nd::all) = (A(nd::all, -1, nd::all) + B(nd::all, nd::all, -1)) / 2.;
```

Alternatively arbitrary functions can be broadcasted to an arbitrary number of tensors of the same 
shape with the `broadcast` and `broadcastIndex` functors:

```
nd::NDArray<float, 3> A(3,3,3), B(3,3,3), C(3,3,3);
/* A = ..., B = ...;  initialize values. */
// Apply C = A*A - B elementwise.
nd::broadcast([](float a, float b, float& c){ c = a * a + b;}, A, B, C);

```

```
// Use the value of the indices (stored in a std::array) to perform operations.
nd::broadcastIndex([](float& a, auto& index){ a = index[0] + index[1] + index[2];}, A);
```
