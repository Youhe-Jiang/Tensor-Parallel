# import ctypes
# ll = ctypes.cdll.LoadLibrary
# lib = ll('./test_cpp.so')
# lib.test.argtypes = [
#     ctypes.c_int,
#     ctypes.c_int,
#     ctypes.c_int, 
# ]
# lib.test(100,200,300)

# import ctypes
# import numpy as np

# mylib = ctypes.CDLL('./test_cpp.so')

# mylib.my_cpp_function.argtypes = [
#     ctypes.c_int,
#     np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),
#     ctypes.c_int,
#     ctypes.c_int,
# ]

# mylib.my_cpp_function.restype = None

# def call_cpp_function(a, b):
#     rows, cols = b.shape
#     b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
#     mylib.my_cpp_function(a, b_ptr, rows, cols)
#     return b

# a = 2
# b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
# result = call_cpp_function(a, b)
# print(result)

# Python code
import mylib
import numpy as np

layer_num = 2
max_mem = 3
strategy_num = 4

v_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
_mark = np.zeros((layer_num, max_mem, strategy_num), dtype=np.int32)
_f = np.zeros((max_mem, strategy_num))
inter_cost = np.random.random((layer_num, strategy_num, strategy_num))
intra_cost = np.random.random((layer_num, strategy_num))
res_list = np.zeros(layer_num, dtype=np.int32)

result = mylib.dynamic_programming_function(layer_num, max_mem, strategy_num, v_data, _mark, _f, inter_cost, intra_cost, res_list)

print("Total Cost:", result[0])
print("Remaining Memory:", result[1])
print("Res List:", res_list)