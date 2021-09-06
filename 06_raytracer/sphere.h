#ifndef SPHERE_H
#define SPHERE_H

#include <cmath>
#include <limits>

// Need to evaluate this here instead of the __device__ function
// Avoids calling a constexpr __host__ function from a __device__ function is
// not allowed
constexpr double lowest{std::numeric_limits<double>::lowest()};

struct Sphere {
    // Check if ray shoot from pixel (ox, oy) hits the sphere
    __host__ __device__ double hit(double ox, double oy, double* n) {
        // A ray is shoot from pixel at position (ox, oy)
        // Ray travels only along z axis and the camera is at the origin

        // Compute distance on the (x, y) plane from the pixel to the sphere
        // center
        double dx = ox - x;
        double dy = oy - y;

        double dx2 = dx * dx;
        double dy2 = dy * dy;
        double radius2 = radius * radius;

        // Check if ray from pixel (ox, oy) intercepts the sphere (cross
        // section)
        if (dx2 + dy2 < radius2) {
            // Compute distance from the camera where the ray hits the sphere
            // From the center of the sphere
            double dz{std::sqrt(radius2 - dx2 - dy2)};
            *n = dz / std::sqrt(radius2);

            // Return actual distance travelled by the ray along z (before
            // hitting)
            return dz + z;
        }

        // Ray does not hit anything and travels to -infinity along z
        return lowest;
    }

    double r, g, b;
    double radius;
    double x, y, z;
};

#endif