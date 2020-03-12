#!/bin/sh
g++ `pkg-config --cflags gsl` -o matrix_inverse.exe \
    matrix_inverse.cpp `pkg-config --libs gsl`;

