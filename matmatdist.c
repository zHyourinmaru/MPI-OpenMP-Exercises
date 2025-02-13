




#include <omp.h>
#include <mpi.h>
#include <stdlib.h>


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




void matmatdist(MPI_Comm Gridcom, int LDA, int LDB, int LDC, double *A, double *B, double *C, int N1, int N2, int N3, int DB1, int DB2, int DB3, int NTrow, int NTcol){

    int coords[2], griddims[2], gridperiods[2], coldir[2], rowdir[2];
    int a, b, quotient, rest, mcm;
    int rowA, colArowB, colB, Browdim, Acoldim;
    int elementindex, i, j, k, colindex, rowindex;
    double *Acol, *Brow, *Aptr, *Bptr;

    MPI_Comm rowcomm, colcomm;

    MPI_Cart_get(Gridcom, 2, griddims, gridperiods, coords);

    rowdir[0] = 0;
    rowdir[1] = 1;

    MPI_Cart_sub(Gridcom, rowdir, &rowcomm);

    coldir[0] = 1;
    coldir[1] = 0;

    MPI_Cart_sub(Gridcom, coldir, &colcomm);

    a = griddims[0];
    b = griddims[1];

    while (b != 0) {
        quotient = a/b;
        rest = a - quotient * b;
        a = b;
        b = rest;
    }

    mcm = griddims[0] * griddims[1] / a;

    rowA = N1/griddims[0];
    colB = N3/griddims[1];
    colArowB = N2/mcm;

    Acoldim = rowA * colArowB;
    Browdim = colArowB * colB;
    Acol = (double*) malloc (Acoldim * sizeof(double));
    Brow = (double*) malloc (Browdim * sizeof(double));

    Aptr = A;
    Bptr = B;

    for (k = 0; k < mcm; k++) {

        colindex = k % griddims[1];
        rowindex = k % griddims[0];

        if (coords[1] == colindex) {

            elementindex = 0;

            for (i = 0; i < rowA; i++) {
                for (j = 0; j < colArowB; j++) {
                    Acol[elementindex] = Aptr[i * LDA + j];
                    elementindex++;
                }
            }

            Aptr = Aptr + colArowB;

        }

        if (coords[0] == rowindex) {

            elementindex = 0;

            for (i = 0; i < colArowB; i++){
                for (j = 0; j < colB; j++) {
                    Brow[elementindex] = Bptr[i * LDB + j];
                    elementindex++;
                }
            }

            Bptr = Bptr + (colArowB * LDB); 

        }

        MPI_Bcast(Acol, Acoldim, MPI_DOUBLE, colindex, rowcomm);
        MPI_Bcast(Brow, Browdim, MPI_DOUBLE, rowindex, colcomm);

        matmatthread(colArowB, colB, LDC, Acol, Brow, C, rowA, colArowB, colB, DB1, DB2, DB3, NTrow, NTcol);

    }

    free(Acol);
    free(Brow);

}
