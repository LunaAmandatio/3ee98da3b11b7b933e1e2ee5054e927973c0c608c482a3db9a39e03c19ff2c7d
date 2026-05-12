PWD = $(shell pwd)
all:lib dgemmtest hgemmtest

lib:
	g++ src/gemm.cpp -shared -o libgemm.so -fPIC -I ./include/ -fopenmp -O3
dgemmtest: lib
	g++ test/gemmtest.cpp test/help.cpp -I ./include/ -I ./test/ -lkblas -L $(PWD) -lgemm -o dgemmbench -DFLOAT64
hgemmtest: lib
	g++ test/gemmtest.cpp test/help.cpp -I ./include/ -I ./test/ -lkblas -L $(PWD) -lgemm -o hgemmbench -DFLOAT16
clean:
	rm -rf *.so *.o test/*.o dgemmbench hgemmbench


noKBLS:noKBLS_lib noKBLS_dgemmtest

noKBLS_dgemmtest:test/gemmtest.cpp noKBLS/kblas.h
	g++ test/gemmtest.cpp test/help.cpp -I ./include/ -I ./test/ -I ./noKBLS/ -L $(PWD) -lgemm -o dgemmbench -DFLOAT64 -DnoKBLS

noKBLS_hgemmtest:test/gemmtest.cpp noKBLS/kblas.h
	g++ test/gemmtest.cpp test/help.cpp -I ./include/ -I ./test/ -I ./noKBLS/ -L $(PWD) -lgemm -o hgemmbench -DFLOAT16 -DnoKBLS

noKBLS_lib:src/gemm.cpp include/gemm.h noKBLS/kblas.h
	g++ src/gemm.cpp -shared -o libgemm.so -fPIC -I ./include/ -I ./noKBLS/ -fopenmp -O3 -DnoKBLS
