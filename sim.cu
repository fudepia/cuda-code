#include<bits/stdc++.h>
#include"cls.hpp"

using namespace std;



__global__ void readFirst(coord<int> *x) {
    printf("CUDA* %d\n", (*x).d[0]);
}

void readFirstC(coord<int> *x) {
    printf("%d\n", (*x).d[0]);
}
int main() {
    coord<int> a={2,0,2,0,1,2,2,4};
    coord<int> ts={1,2,2,4};
    coord<int> *ah=&a,*bd;
    coord<int> test(a);
    auto nB=sizeof(a);
    cout << a<< endl;
    cout << test << endl;
    cout << *ah << endl;
    cudaMalloc((void **) &bd, nB);
    coord<int> *ad=(*ah).toGPU();
    cout  << "──────────────────────" << endl;




    readFirstC(ah);
    readFirst<<<1,1>>>(ad);
    cudaDeviceSynchronize();
    cudaMemcpy(bd, ad, nB, cudaMemcpyDeviceToDevice); //Need work, shallow copy
    readFirst<<<1,1>>>(bd);
    cudaDeviceSynchronize();
    coord<int> *bh=(*ad).pull();
    readFirstC(bh);
//    for(di i=0;i<8;i++) cout << *((*bh).d+i) << ", ";
//    cout << endl;
    //cout << *((*bh).d) << endl;
    //dim3 dimBlock();
    return 0;
}
