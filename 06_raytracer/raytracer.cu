#include <cstddef>
#include <cassert>
#include <iostream>
#include <limits>

#include "sphere.h"

void raytracer_cpu(char* image, Sphere* s, std::size_t width,
                   std::size_t height, std::size_t num_spheres) {
    for (std::size_t x{0}; x < width; x++) {
        for (std::size_t y{0}; y < height; y++) {
            std::size_t offset = x + y * width;

            double ox{x - width / 2.0};
            double oy{y - height / 2.0};

            double r{0.0}, g{0.0}, b{0.0};
            double maxz{lowest};

            for (std::size_t i{0}; i < num_spheres; i++) {
                double n{0.0};

                // Determine if/where ray from pixel (ox, oy) hits a sphere
                double zhit = s[i].hit(ox, oy, &n);

                // Check if current pixel hit sphere closer than previous spere
                if (zhit > maxz) {
                    r = s[i].r * n;
                    g = s[i].g * n;
                    b = s[i].b * n;
                    maxz = zhit;
                }
            }

            // C-style casting of colors to size_t
            image[offset * 3 + 0] = (std::size_t)(r * 255);
            image[offset * 3 + 1] = (std::size_t)(g * 255);
            image[offset * 3 + 2] = (std::size_t)(b * 255);
        }
    }
}

__global__ void raytracer_kernel(char* image, Sphere* s, std::size_t width,
                                 std::size_t height, std::size_t num_spheres) {
    std::size_t x{threadIdx.x + blockIdx.x * blockDim.x};
    std::size_t y{threadIdx.y + blockIdx.y * blockDim.y};
    std::size_t offset{x + y * blockDim.x * gridDim.x};

    double ox{x - width / 2.0};
    double oy{y - height / 2.0};

    double r{0.0}, g{0.0}, b{0.0};
    double maxz{lowest};

    for (std::size_t i{0}; i < num_spheres; i++) {
        double n{0.0};

        // Determine if/where ray from pixel (ox, oy) hits a sphere
        double zhit = s[i].hit(ox, oy, &n);

        // Check if current pixel hit sphere closer than previous spere
        if (zhit > maxz) {
            r = s[i].r * n;
            g = s[i].g * n;
            b = s[i].b * n;

            maxz = zhit;
        }
    }

    // C-style casting of colors to size_t
    image[offset * 3 + 0] = (std::size_t)(r * 255);
    image[offset * 3 + 1] = (std::size_t)(g * 255);
    image[offset * 3 + 2] = (std::size_t)(b * 255);
}
