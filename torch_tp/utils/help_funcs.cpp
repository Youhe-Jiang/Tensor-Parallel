void print_array1d_int(int* ptr, int x){
    for (int i = 0; i < x; ++i) {
        std::cout << ptr[i] << ' ';
        std::cout << std::endl;
    }
}

void print_array1d_double(double* ptr, int x){
    for (int i = 0; i < x; ++i) {
        std::cout << ptr[i] << ' ';
        std::cout << std::endl;
    }
}

void print_array1d_vector(std::vector<double> ptr, int x){
    for (int i = 0; i < x; ++i) {
        std::cout << ptr[i] << ' ';
        std::cout << std::endl;
    }
}

void print_array2d_int(int* ptr, int x, int y){
    for (int i = 0; i < x; ++i) {
        for (int s = 0; s < y; ++s) {
            std::cout << ptr[i * y + s] << ' ';
        }
        std::cout << std::endl;
    }
}
void print_array2d_double(double* ptr, int x, int y){
    for (int i = 0; i < x; ++i) {
        for (int s = 0; s < y; ++s) {
            std::cout << ptr[i * y + s] << ' ';
        }
        std::cout << std::endl;
    }
}
