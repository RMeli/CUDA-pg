#ifndef NUM_H
#define NUM_H

template <typename T>
bool nearly_equal(T a, T b, double rtol=1e-05, double atol=1e-08){
    bool nequal{false};

    if(std::abs(a - b) <= atol + rtol * std::abs(b)){
        nequal = true;
    }

    return nequal;
}

#endif // NUM_H