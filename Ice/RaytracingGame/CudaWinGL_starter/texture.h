#ifndef TEXTURE_H
#define TEXTURE_H

#include "color.h"
#include <memory>

using namespace std;

enum texture_type
{
    Solid,
    Checker,
};

class texture {
public:
    __device__ texture(color first)
        : first(first)
        , type(Solid)
    {
    }

    __device__ texture(color first, color second)
        : first(first)
        , second(second)
        , type(Checker)
    {
    }

    __device__ color value(float u, float v, const point3& p) const
    {
        if (type == texture_type::Solid)
            return first;
        else if (type == texture_type::Checker)
        {
            auto u2 = int(std::floor(u * 25.0f));
            auto v2 = int(std::floor(v * 25.0f));

            return (u2 + v2) % 2 == 0 ? first : second;
        }

        return first;
    }

    int type;
    color first;
    color second;
};

/*
class solid_color : public texture {
public:
    __device__ solid_color(const color& albedo) : albedo(albedo) {}

    __device__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

    __device__ color value(float u, float v, const point3& p) const override {
        return albedo;
    }

private:
    color albedo;
};

class checker_texture : public texture {
public:
    __device__ checker_texture(float scale, texture* even, texture* odd)
        : inv_scale(1.0 / scale), even(even), odd(odd) {}

    __device__ checker_texture(float scale, const color& c1, const color& c2)
        : checker_texture(scale, new solid_color(c1), new solid_color(c2)) {}

    __device__ virtual ~checker_texture()
    {
        printf("CCCCC");
        delete even;
        delete odd;
    }

    __device__ color value(float u, float v, const point3& p) const override {
        auto u2 = int(std::floor(u * inv_scale));
        auto v2 = int(std::floor(v * inv_scale));

        return (u2 + v2) % 2 == 0 ? even->value(u, v, p) : odd->value(u, v, p);

    }

private:
    float inv_scale;
    texture* even = nullptr;
    texture* odd = nullptr;
};
*/

#endif