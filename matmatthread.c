
#include <omp.h>

void matmatijk (int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3) {
    int i, j, k;
    
    for (i = 0; i < N1; i++) {
        for (j = 0; j < N3; j++) {
            for (k = 0; k < N2; k++) {
                C[i * ldC + j] = C[i * ldC + j] + A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

void matmatikj (int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3) {
    int i, j, k;
    
    for (i = 0; i < N1; i++) {
        for (k = 0; k < N2; k++) {
            for (j = 0; j < N3; j++) {
                C[i * ldC + j] = C[i * ldC + j] + A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

void matmatkij (int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3) {
    int i, j, k;
    
    for (k = 0; k < N2; k++) {
        for (i = 0; i < N1; i++) {
            for (j = 0; j < N3; j++) {
                C[i * ldC + j] = C[i * ldC + j] + A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

void matmatkji (int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3) {
    int i, j, k;
    
    for (k = 0; k < N2; k++) {
        for (j = 0; j < N3; j++) {
            for (i = 0; i < N1; i++) {
                C[i * ldC + j] = C[i * ldC + j] + A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

void matmatjik (int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3) {
    int i, j, k;
    
    for (j = 0; j < N3; j++) {
        for (i = 0; i < N1; i++) {
            for (k = 0; k < N2; k++) {
                C[i * ldC + j] = C[i * ldC + j] + A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

void matmatjki (int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3) {
    int i, j, k;
    
    for (j = 0; j < N3; j++) {
        for (k = 0; k < N2; k++) {
            for (i = 0; i < N1; i++) {
                C[i * ldC + j] = C[i * ldC + j] + A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}

void matmatblock (int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3, int dbA, int dbB, int dbC) {

    int i, j, k;
    
    for (i = 0; i < N1/dbA; i++) {
        for (j = 0; j < N3/dbC; j++) {
            for (k = 0; k < N2/dbB; k++) {
                matmatikj(ldA, ldB, ldC, &A[(i * ldA + k) * dbA], &B[(k * ldB + j) * dbB], &C[(i * ldC + j) * dbC], dbA, dbB, dbC);
            }
        }
    }

}



void matmatthread(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N1, int N2, int N3, int dbA, int dbB, int dbC, int NTROW, int NTCOL) {
    
    int NT = NTROW * NTCOL, id, idi, idj, starti, startj;

    omp_set_num_threads(NT);

    #pragma omp parallel private(id, idi, idj, starti, startj)
    {
        id = omp_get_thread_num();
        idi = id / NTCOL;         
        idj = id % NTCOL;          

        starti = idi * (N1 / NTROW);
        startj = idj * (N3 / NTCOL);

        matmatblock(ldA, ldB, ldC, &A[starti * ldA], &B[startj], &C[starti * ldC + startj], N1 / NTROW, N2, N3 / NTCOL, dbA, dbB, dbC);               
    }
}