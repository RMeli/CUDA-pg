#ifndef RAYTRACER_H
#define RAYTRACVER_H

#include <cstddef>

#include "sphere.h"

void raytracer_cpu(char* image, Sphere* s, std::size_t width,
                   std::size_t height, std::size_t num_spheres);

__global__ void raytracer_kernel(char* image, Sphere* s, std::size_t width,
                                 std::size_t height, std::size_t num_spheres);

#endif