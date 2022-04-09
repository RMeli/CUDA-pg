#ifndef ANIMATION_H
#define ANIMATION_H

#include <string>

#include <GL/glext.h>
#include <GL/glut.h>
#include <GL/glx.h>

auto get_address(std::string str);

class Animation {
  public:
    Animation(int w, int h, void* d = nullptr);
    ~Animation();

    unsigned char* image();
    std::size_t size();

    void animate(void (*f)(void*, int));

    static Animation** bitmap();

  private:
    unsigned char* pixels;
    int width, height;
    void* data;
};

#endif // ANIMATION_H