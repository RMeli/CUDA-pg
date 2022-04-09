
#include "animation.h"

auto get_address(std::string str){
    return glXGetProcAddress((const GLubyte *) str.c_str());
}


Animation::Animation(int w, int h, void* d)
: width(w), height(h), data(d)
{
    pixels = new unsigned char[width * height * 4];
}

Animation::~Animation(){
    delete[] pixels;
}

unsigned char* Animation::image(){
    return pixels;
}

std::size_t Animation::size(){
    return width * height * 4;
}

void Animation::animate(void (*f)(void*,int)){
    Animation** bmp = bitmap();
    *bmp = this;

    int c{0};
    char* dummy{nullptr};
    glutInit(&c, &dummy);
}

Animation** Animation::bitmap(){
    static Animation* gbitmap;
    return &gbitmap;
}
