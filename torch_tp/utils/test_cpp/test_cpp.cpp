// #include <iostream>
// #include <string>
// using namespace std;
// extern "C"{
//     void test(int n1, int n2, int n3){
//             cout<<"The C++ input 1 is "<<n1<<"\n";
//             cout<<"The C++ input 2 is "<<n2<<"\n";
//             cout<<"The C++ input 3 is "<<n3<<"\n";
//         }
// }

// extern "C" {
//     void my_cpp_function(int a, int** b, int rows, int cols) {
//         for (int i = 0; i < rows; ++i) {
//             for (int j = 0; j < cols; ++j) {
//                 b[i][j] = a * b[i][j];
//             }
//         }
//     }
// }


// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// namespace py = pybind11;

// void my_cpp_function(int a, py::array_t<int> b) {
//     py::buffer_info info = b.request();
//     int *ptr = static_cast<int *>(info.ptr);
//     size_t rows = info.shape[0];
//     size_t cols = info.shape[1];

//     for (size_t i = 0; i < rows; ++i) {
//         for (size_t j = 0; j < cols; ++j) {
//             ptr[i * cols + j] = a * ptr[i * cols + j];
//         }
//     }
// }

// PYBIND11_MODULE(mylib, m) {
//     m.def("my_cpp_function", &my_cpp_function, "A function that takes an int and a 2D numpy array");
// }


// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// namespace py = pybind11;

// void complex_cpp_function(int layer_num, int max_mem, int strategy_num,
//                           py::array_t<int> v_data,
//                           py::array_t<int> _mark,
//                           py::array_t<double> _f,
//                           py::array_t<double> inter_cost,
//                           py::array_t<double> intra_cost,
//                           py::array_t<int> res_list) {

//     py::buffer_info v_data_info = v_data.request();
//     int* v_data_ptr = static_cast<int*>(v_data_info.ptr);

//     py::buffer_info _mark_info = _mark.request();
//     int* _mark_ptr = static_cast<int*>(_mark_info.ptr);

//     py::buffer_info _f_info = _f.request();
//     double* _f_ptr = static_cast<double*>(_f_info.ptr);

//     py::buffer_info inter_cost_info = inter_cost.request();
//     double* inter_cost_ptr = static_cast<double*>(inter_cost_info.ptr);

//     py::buffer_info intra_cost_info = intra_cost.request();
//     double* intra_cost_ptr = static_cast<double*>(intra_cost_info.ptr);

//     py::buffer_info res_list_info = res_list.request();
//     int* res_list_ptr = static_cast<int*>(res_list_info.ptr);

//     // Your complex computation here...
//     // You can access the data using the pointers and perform your calculations
//     // based on the provided input arguments.

//     // For demonstration purposes, let's just print some values from the input arrays.
//     for (int i = 0; i < layer_num; ++i) {
//         for (int j = 0; j < strategy_num; ++j) {
//             int v_value = v_data_ptr[i * strategy_num + j];
//             int mark_value = _mark_ptr[i * max_mem * strategy_num + j];
//             double f_value = _f_ptr[j];
//             double inter_cost_value = inter_cost_ptr[i * strategy_num * strategy_num + j * strategy_num + j];
//             double intra_cost_value = intra_cost_ptr[i * strategy_num + j];
//             int res_value = res_list_ptr[i];

//             printf("Layer %d, Strategy %d:\n", i, j);
//             printf("  v_data: %d\n", v_value);
//             printf("  _mark: %d\n", mark_value);
//             printf("  _f: %f\n", f_value);
//             printf("  inter_cost: %f\n", inter_cost_value);
//             printf("  intra_cost: %f\n", intra_cost_value);
//             printf("  res_list: %d\n", res_value);
//         }
//     }
// }

// PYBIND11_MODULE(mylib, m) {
//     m.def("complex_cpp_function", &complex_cpp_function, "A complex function with multiple input arrays");
// }



#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <limits>

namespace py = pybind11;

std::pair<double, int> dynamic_programming_function(int layer_num,
                                                    int max_mem,
                                                    int strategy_num,
                                                    py::array_t<int> v_data,
                                                    py::array_t<int> _mark,
                                                    py::array_t<double> _f,
                                                    py::array_t<double> inter_cost,
                                                    py::array_t<double> intra_cost,
                                                    py::array_t<int> res_list) {

    py::buffer_info v_data_info = v_data.request();
    int* v_data_ptr = static_cast<int*>(v_data_info.ptr);

    py::buffer_info _mark_info = _mark.request();
    int* _mark_ptr = static_cast<int*>(_mark_info.ptr);

    py::buffer_info _f_info = _f.request();
    double* _f_ptr = static_cast<double*>(_f_info.ptr);

    py::buffer_info inter_cost_info = inter_cost.request();
    double* inter_cost_ptr = static_cast<double*>(inter_cost_info.ptr);

    py::buffer_info intra_cost_info = intra_cost.request();
    double* intra_cost_ptr = static_cast<double*>(intra_cost_info.ptr);

    py::buffer_info res_list_info = res_list.request();
    int* res_list_ptr = static_cast<int*>(res_list_info.ptr);

    for (int i = 0; i < layer_num; ++i) {
        for (int v = max_mem - 1; v >= 0; --v) {
            for (int s = 0; s < strategy_num; ++s) {
                if (v < v_data_ptr[i * strategy_num + s]) {
                    _mark_ptr[i * max_mem * strategy_num + v * strategy_num + s] = -1;
                    _f_ptr[v * strategy_num + s] = std::numeric_limits<double>::infinity();
                    continue;
                }
                std::vector<double> candidates(strategy_num);
                for (int si = 0; si < strategy_num; ++si) {
                    candidates[si] = _f_ptr[(v - v_data_ptr[i * strategy_num + s]) * strategy_num + si] + inter_cost_ptr[i * strategy_num * strategy_num + si * strategy_num + s];
                }

                double min_value = candidates[0];
                int min_index = 0;
                for (int ci = 1; ci < strategy_num; ++ci) {
                    if (candidates[ci] < min_value) {
                        min_value = candidates[ci];
                        min_index = ci;
                    }
                }

                _mark_ptr[i * max_mem * strategy_num + v * strategy_num + s] = min_index;
                _f_ptr[v * strategy_num + s] = min_value + intra_cost_ptr[i * strategy_num + s];
            }
        }
    }

    int next_index = 0;
    int next_v = max_mem - 1;
    double total_cost = _f_ptr[(layer_num - 1) * strategy_num + next_index];

    if (!(total_cost < std::numeric_limits<double>::infinity())) {
        return {std::numeric_limits<double>::infinity(), -1};
    }

    res_list_ptr[layer_num - 1] = next_index;

    for (int i = layer_num - 2; i >= 0; --i) {
        next_index = _mark_ptr[i * max_mem * strategy_num + next_v * strategy_num + next_index];
        next_v -= v_data_ptr[i * strategy_num + next_index];
        res_list_ptr[i] = next_index;
    }

    return {total_cost, next_v - v_data_ptr[0 * strategy_num + next_index]};
}

PYBIND11_MODULE(mylib, m) {
    m.def("dynamic_programming_function", &dynamic_programming_function, "A dynamic programming function");
}
