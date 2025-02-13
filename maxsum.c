#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double maxsum(int N, int LD, double* A, int NT) {

    int id, start, end, i, j;
    double MAX = 0, sum, maxrow;

    omp_set_num_threads(NT);

    #pragma omp parallel private (id, start, end, sum, i, j, maxrow)
    {

        id = omp_get_thread_num();
        start = id * N/NT;
        end = (id + 1) * N/NT;
        maxrow = 0;

        for (i = start; i < end; i++) {
            
            sum = 0;
            
            for (j = 0; j < N; j++) {
                    sum = sqrt(A[i*LD+j]) + sum;
            }

            if (sum > maxrow) {
                maxrow = sum;
            }
            
        }

        #pragma omp critical
        {
            if (maxrow > MAX) {
                MAX = maxrow;
            }
        }

    }

    return MAX;

}