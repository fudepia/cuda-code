// https://qiita.com/agate-pris/items/1b29726935f0b6e75768
// https://qiita.com/MoriokaReimen/items/7c83ebd0fbae44d8532d
#ifndef   CLS_HPP
#define   CLS_HPP


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//https://stackoverflow.com/a/14038590/8460574

#include<bits/stdc++.h>
using namespace std;
#define dt 1e-20
#define RES 3
#define SL 1000000 //sideLength
#define angIncr 1e-12
#define DxIncr 1e-4 //must be way smaller than resolution
#define di long unsigned int
#define ull unsigned long long int
#define deg unsigned short

#ifdef __CUDA_ARCH__
#define CUDA_CALLABLE_MEMBER __device__
#define MEMCPY memcpy
#define MEMCPYMETHOD cudaMemcpyDeviceToDevice
#else
#define CUDA_CALLABLE_MEMBER
#define MEMCPY cudaMemcpy
#define MEMCPYMETHOD cudaMemcpyHostToHost
#endif

template<typename T>
struct coord {
    T *d;
    long unsigned int dim;
    //coord(int dim, T val=0.0): d(dim, val), dim(dim) {}
    CUDA_CALLABLE_MEMBER coord(di dim, T val=0.0): dim(dim) {
        d=new T[dim];
        for(di i=0;i<dim;i++) {
            d[i]=val;
        }
    }
    CUDA_CALLABLE_MEMBER coord(double r, deg theta, deg phi, double divisor=1.0) { // theta<-[0,360), phi<-[0,180]
        dim=3;
        double t=(double)theta/divisor/180*M_PI, p=(double)phi/divisor/180*M_PI;
        d={r*cos(t)*sin(p),r*sin(t)*sin(p),r*cos(p)};
    }
    CUDA_CALLABLE_MEMBER coord(double r, deg theta, double divisor=1.0) {
        dim=2;
        double t=(double)theta/divisor/180*M_PI;
        d={r*cos(t), r*sin(t)};
    }
    CUDA_CALLABLE_MEMBER coord(initializer_list<T> l): dim{l.size()} {
        //d=(T *)calloc(dim, sizeof(T));
        d=new T[dim];
        memcpy(d, l.begin(), sizeof(T)*dim);
        //std::copy(l.begin(), l.end(), d);
    }
    CUDA_CALLABLE_MEMBER ~coord() {
        #ifdef __CUDA_ARCH__
        //delete[]d;
        free(d);
        #else
        cout << "dsrt: "<< d << endl;
        //delete[]d;
        //gpuErrchk(cudaFree(d));// GPUassert: invalid argument cls.hpp 72  
        #endif
    }
    CUDA_CALLABLE_MEMBER coord(const coord<T> &other): dim(other.dim) {
        printf("copy ro3");
        d=new T[dim];
        memcpy(d, other.d, sizeof(T)*dim);
    }
    CUDA_CALLABLE_MEMBER coord(coord<T> &&other) noexcept: d(std::exchange(other.d, nullptr)), dim(std::exchange(other.dim,(di)0)) {}
    CUDA_CALLABLE_MEMBER coord& operator=(const coord<T> &other) {
        printf("move ro3");
        if (this == &other) return *this;
        dim=other.dim;
        d=new T[dim];
        memcpy(d, other.d, sizeof(T)*dim);
    }
    CUDA_CALLABLE_MEMBER coord& operator=(coord<T> &&other) noexcept {
        swap(dim, other.dim);
        swap(d, other.d);
        return *this;
    }
    coord<T>* toGPU() {
        auto cpy=*this;
        coord<T> *dest;
        auto nB=sizeof(cpy);
        auto dB=sizeof(sizeof(T)*dim);
        gpuErrchk(cudaMalloc((void **) &dest, nB));
        gpuErrchk(cudaMalloc((void **) &(cpy.d), nB));
        gpuErrchk(cudaMemcpy((cpy.d), (*this).d, dB, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dest, &cpy, nB, cudaMemcpyHostToDevice));
        cout << "Cpy: " << cpy.d << endl;
        return dest;
    }
    coord<T>* pull() {
        auto nB=sizeof(*this);
        auto dB=sizeof(sizeof(T)*dim);
        coord<T> *dest=(coord<T> *)malloc(nB);
        gpuErrchk(cudaMemcpy(dest, this, nB, cudaMemcpyDeviceToHost));
        //gpuErrchk(cudaMemcpy(&((*dest).dim), &((*this).dim), sizeof(di), cudaMemcpyDeviceToHost));
        auto pBk=(*dest).d;
        (*dest).d=new T[(*dest).dim];
        cudaMemcpy((*dest).d, pBk, dB, cudaMemcpyDeviceToHost);
        return dest;
    }
    double norm() const {
        double res=0;
        for(di i=0; i<dim; i++) {
            res+=pow(this->d[i],2);
        }
        return sqrt(res);
    }
    coord<T> unit() {
        return *this/this->norm();
    }
    coord<int> round() const {
        coord<int> crd(dim, 0);
        for(int i=0;i<dim;i++) {
            crd[i]+=std::round(d[i]);
        }
        return crd;
    }
    coord<T> abs() {
        coord<T> res(this->dim);
        for(di i=0; i<this->dim; i++) { res[i]=std::abs(this->d[i]); }
        return res;
    }
    coord<ull> unsign() const {
        coord<ull> crd(dim, (ull)0);
        for(int i=0;i<dim;i++) {
            crd[i]=(ull)std::abs(std::round(d[i]));
        }
        return crd;
    }
    T dot(coord<T> &r) const {
        auto res=*this;
        for(di i=0; i<r.dim; i++) { res.d[i]*=r[i]; };
        double ans=0;
        for(auto &i:res.d) ans+=i;
        return ans;
    }
//    void loopTo(coord<T> dest, void(*f)(coord<T> x)) const {
//        dest-=*this;
//        //field tmp(coord(dest.dim), dest);
//        int end=field::coordZuID(dest, dest.d);
//        for(int i=0; i<=end; i++) {
//            (*f)(field::IDzuCoord(i, dest.dim, dest.d)+*this);
//        }
//        return;
//    }
    coord<T>& operator+=(const coord<T> &r) {
        //for(int i=0; i<(r.dim<this->dim?r.dim:this->dim); i++) {
        if(r.dim>this->dim) {
            throw "Addition's rhs has larger Dimension than lhs.";
        }
        for(di i=0; i<r.dim; i++) { this->d[i]+=r[i]; }
        return *this;
    }
    coord<T> operator-=(const coord<T> &r) { for(di i=0; i<r.dim; i++) { this->d[i]-=r[i]; }; return *this; }
    coord<T> operator*=(const T &r) { for(di i=0; i<this->dim; i++) { this->d[i]*=r; }; return *this; }
    coord<T> operator/=(const T &r) { for(di i=0; i<this->dim; i++) { this->d[i]/=r; }; return *this; }
    CUDA_CALLABLE_MEMBER T& operator[](int i) {
        return d[i];
    }
    CUDA_CALLABLE_MEMBER T operator[](int i) const {
        return d[i];
    }
    friend ostream& operator<<(ostream& os, const coord<T> &crd) {
        os << crd[0];
        for(di i=1;i<crd.dim;i++) {
            os << ", " << crd[i];
        }
        return os;
    }
    friend coord<T> operator+(coord<T> lhs, const coord<T> &r) { return lhs+=r; }
    friend coord<T> operator-(coord<T> lhs, const coord<T> &r) { return lhs-=r; }
    friend coord<T> operator*(coord<T> lhs, const T &r) { return lhs*=r; }
    friend coord<T> operator/(coord<T> lhs, const T &r) { return lhs/=r; }
};

template<typename G>
struct mono {
    coord<G> pos,vel;
    mono(di dim=3) {
        coord<G> pos(dim);
        coord<G> vel(dim);
    }
};

template<typename H>
struct field {
    H* d; // [-SL,SL]*[-SL,SL]
    ull border; //max ID
    ull *sL;
    coord<int> *baseCrd;
    field(coord<H> bl, coord<H> tr, H val=1) {
        baseCrd=new coord<int>(bl.round());
        sL=new ull[(*baseCrd).dim];
        auto tmp=(tr-bl).unsign();
        for(di i=0;i<bl.dim;i++) {
            sL[i]=tmp[i]+1;
        }
        border=coordZuID(tr);
        //init d
        d=(H*)calloc(border+1, sizeof(*d));
        // https://stackoverflow.com/a/29977424/8460574
        for(ull i=0;i<=border;i++) d[i]=val;
    }
    field(const coord<H> sz,H val=1):field(sz*-1, sz, val){}
    ~field() {
        delete[]sL;
        delete[]d;
        (*baseCrd).~coord();
    }
    field(const field<H> &other): border(other.border) {
        baseCrd=new coord<int>(*(other.baseCrd));
        sL=new ull[(*baseCrd).dim];
        for(di i=0;i<(*baseCrd).dim;i++) {
            sL[i]=other.sL[i];
        }
        d=(H*)calloc(border+1, sizeof(*d));
        // https://stackoverflow.com/a/29977424/8460574
        for(ull i=0;i<=border;i++) d[i]=other[i];
    }
    //field(const field<H> &&other) noexcept: d(std::exchange(other.d, nullptr)), border, sL, baseCrd {}
    field& operator=(const field<H> &other) {
        if (this == &other) return *this;
        baseCrd=new coord<int>(*(other.baseCrd));
        sL=new ull[(*baseCrd).dim];
        for(di i=0;i<(*baseCrd).dim;i++) {
            sL[i]=other.sL[i];
        }
        d=(H*)calloc(border+1, sizeof(*d));
        // https://stackoverflow.com/a/29977424/8460574
        for(ull i=0;i<=border;i++) d[i]=other[i];
    }
    //field& operator=(field<H> &&other) noexcept {
    ull coordZuID(coord<H> x) {
        ull id=0;
        auto cvrtCrd=x.round()-(*baseCrd);
        for(di i=0; i<x.dim; i++) {
            ull inRes=std::round(cvrtCrd[i]);
            di j=i;
            while(j!=0) {
                j--;
                inRes*=sL[j];
            }
            id+=inRes;
        }
        return id;
    }
    coord<int> IDzuCoord(int id) {
        di i=(*baseCrd).dim;
        coord<int> x(i);
        do {
            i--;
            int b=pow(sL[i], i);
            x[i]=id/b;
            id-=x[i]*b;
        } while(i!=0);
        return x;
    }
//    ull coordZuID(coord<H> x, ull sideLength) { return coordZuID(x.round(), vector<ull>(x.dim, sideLength)); }
//    coord<int> IDzuCoord(int id, di dim, ull sideLength) { return IDzuCoord(id, dim, vector<ull>(dim, sideLength));}
    CUDA_CALLABLE_MEMBER H& operator[](coord<H> x) {
        auto id=field::coordZuID(x.round(), sL);
        return this->d[id>border?border:id];
    }
    CUDA_CALLABLE_MEMBER H& operator[](int flatIndex) {
        return this->d[flatIndex];
        return this->d[flatIndex>border?border:flatIndex];
    }
    CUDA_CALLABLE_MEMBER H& operator[](ull flatIndex) {
        return this->d[flatIndex>border?border:flatIndex];
    }
};

#define nx n[0]
#define ny n[1]
#define nz n[2]
coord<double> R(coord<double> n, double theta, coord<double> target) { //3D only
    double st=sin(theta/180*M_PI), ct=cos(theta/180*M_PI);
    double nx2=pow(nx,2), ny2=pow(ny,2), nz2=pow(nz,2);
    coord<double> res={
        (double)(ct+nx2*(1-ct))*target[0]+(nx*ny*(1-ct)-nz*st)*target[1]+(nz*nx*(1-ct)+ny*st)*target[2],
        (double)(nx*ny*(1-ct)+nz*st)*target[0]+(ct+ny2*(1-ct))*target[1]+(ny*nz*(1-ct)-nx*st)*target[2],
        (double)(nz*nx*(1-ct)-ny*st)*target[0]+(ny*nz*(1-ct)+nx*st)*target[1]+(ct+nz2*(1-ct))*target[2]
    };
    return res;
}

#endif // CLS_HPP
