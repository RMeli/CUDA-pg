#include <fstream>
#include <iostream>


#include "ppm.h"

int main(){

    std::ifstream in("../test.ppm", std::ios::binary);

    if(!in.is_open()){
        std::cout << "FAILED OPENING test.ppm" << std::endl;
        exit(-1);
    }

    // Load image
    char* image{nullptr};
    int width{0}, height{0};
    std::tie(image, width, height) = utils::read_ppm(in);

    std::ofstream out("../out.ppm", std::ios::binary);

    utils::write_ppm(image, width, height, out);

    if(image != nullptr){
        delete[] image;
    }

    return 0;
}