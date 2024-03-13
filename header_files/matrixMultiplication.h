#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VAR1 104
#define VAR2 48
#define VAR3 52

// Function to perform matrix-vector multiplication
void multiplyMatrixVector(float matrix[VAR2][VAR1], float vector[VAR1], float result[VAR2]) {
    for (int i = 0; i < VAR2; i++) {
        result[i] = 0;
        for (int j = 0; j < VAR1; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void multiplyMatrixVector2(float matrix[VAR2][VAR2], float vector[VAR2], float result[VAR2]) {
    for (int i = 0; i < VAR2; i++) {
        result[i] = 0;
        for (int j = 0; j < VAR1; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Function to perform element-wise addition of two vectors
void addVectors(float vector1[VAR2], float vector2[VAR2], float result[VAR2]) {
    for (int i = 0; i < VAR2; i++) {
        result[i] = vector1[i] + vector2[i];
    }
}

// Function to perform element-wise multiplication of two vectors
void multiplyVectors(float vector1[VAR2], float vector2[VAR2], float result[VAR2]) {
    for (int i = 0; i < VAR2; i++) {
        result[i] = vector1[i] * vector2[i];
    }
}










