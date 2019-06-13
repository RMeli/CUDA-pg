#ifndef PPM_H
#define PPM_H

#include <iostream>
#include <string>
#include <tuple>

namespace utils {

std::tuple<char *, int, int> read_ppm(std::istream &in) {
  // check format
  std::string magic_number;
  in >> magic_number;
  if (magic_number != "P6") {
    throw "";
  }

  // get header
  int width, height, max_color;
  in >> width >> height >> max_color;

  // remove withespace
  in.get();

  // get data
  std::size_t size = width * height * 3;
  char *image = new char[size];
  in.read(image, size);

  return std::make_tuple(image, width, height);
}

void write_ppm(char *image, int width, int height, std::ostream &out,
               std::string magic_number = "P6") {
  out << magic_number << std::endl;
  out << width << ' ' << height << std::endl;
  out << 255 << std::endl;

  std::size_t size = width * height * 3;

  out.write(image, size);
}

} // namespace utils

#endif // PPM_H