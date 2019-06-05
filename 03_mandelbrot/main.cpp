#include <fstream>
#include <iostream>

#include "mandelbrot.h"

#include "ppm.h"

int main(){

    std::size_t width{1000}, height{800};

    std::ofstream out("mandelbrot.ppm", std::ios::binary);

    char* image = new char[width * height * 3];
    mandelbrot_serial(image, width, height);

    if(image != nullptr){
        utils::write_ppm(image, width, height, out);
        delete[] image;
    }

    return 0;
}