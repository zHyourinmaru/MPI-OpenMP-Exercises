#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

void laplace(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter) {

    int NP, id, i, j, iter;
    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    for (iter = 0; iter < Niter; iter++) {

        if (id != 0) {
            for (j = 0; j < N; j++) {
                daprev[j] = A[j];
            }

            MPI_Send(daprev, N, MPI_FLOAT, id - 1, 20, MPI_COMM_WORLD);
            MPI_Recv(daprev, N, MPI_FLOAT, id - 1, 10, MPI_COMM_WORLD, &status);
        }

        if (id != NP - 1) {
            for (j = 0; j < N; j++) {
                danext[j] = A[(N/NP - 1) * LD + j];
            }

            MPI_Send(danext, N, MPI_FLOAT, id + 1, 10, MPI_COMM_WORLD);
            MPI_Recv(danext, N, MPI_FLOAT, id + 1, 20, MPI_COMM_WORLD, &status);
        }

        //Se non è il primo processo, calcolo la prima riga
        if (id != 0) {
            for (j = 1; j < N - 1; j++) {
                B[j] = (daprev[j]+ A[1 * LD + j] + A[j - 1] + A[j + 1]) * 0.25;
            }
        }

        //Calcolo la matrice interna ad esclusione della prima ed ultima riga e della prima ed ultima colonna
        for (i = 1; i < N/NP - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                B[(i * LD) + j] = (A[(i + 1) * LD + j]+ A[(i - 1) * LD + j] + A[(i * LD) + (j - 1)] + A[(i * LD) + (j + 1)]) * 0.25;
            }
        }

        //Se non è l'ultimo processo, calcolo l'ultima riga
        if (id != NP - 1) {
            for (j = 1; j < N - 1; j++) {
                B[((N/NP - 1) * LD) + j] = (danext[j]+ A[((N/NP - 1) - 1) * LD + j] + A[((N/NP - 1)  * LD) + (j - 1)] + A[((N/NP - 1)  * LD) + (j + 1)]) * 0.25;
            }
        }

        //Se non è il primo processo, copio la prima riga
        if (id != 0) {
            for (j = 1; j < N - 1; j++) {
                A[j] = B[j];
            }
        }

        //Copio la matrice interna ad esclusione della prima ed ultima riga e della prima ed ultima colonna
        for (i = 1; i < N/NP - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                A[(i * LD) + j] = B[(i * LD) + j];
            }
        }

        //Se non è l'ultimo processo, copio l'ultima riga
        if (id != NP - 1) {
            for (j = 1; j < N - 1; j++) {
                A[((N/NP - 1) * LD) + j] = B[((N/NP - 1) * LD) + j];
            }
        }

    }

    return;
}


void laplace_nb(float *A, float *B, float *daprev, float *danext, int N, int LD, int Niter) {

    int NP, id, i, j, iter;
    MPI_Status status;
    MPI_Request reqs[4];

    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    for (iter = 0; iter < Niter; iter++) {

        if (id != 0) {
            for (j = 0; j < N; j++) {
                daprev[j] = A[j];
            }
            MPI_Isend(daprev, N, MPI_FLOAT, id - 1, 20, MPI_COMM_WORLD, &reqs[0]);
        }

        if (id != NP - 1) {
            for (j = 0; j < N; j++) {
                danext[j] = A[(N / NP - 1) * LD + j];
            }
            MPI_Isend(danext, N, MPI_FLOAT, id + 1, 10, MPI_COMM_WORLD, &reqs[2]);
        }

        if (id != 0) {
            MPI_Wait(&reqs[0], &status);
            MPI_Irecv(daprev, N, MPI_FLOAT, id - 1, 10, MPI_COMM_WORLD, &reqs[1]);
        }

        if (id != NP - 1) {
            MPI_Wait(&reqs[2], &status);
            MPI_Irecv(danext, N, MPI_FLOAT, id + 1, 20, MPI_COMM_WORLD, &reqs[3]);
        }

        // Calcolo la matrice interna ad esclusione della prima ed ultima riga e della prima ed ultima colonna
        for (i = 1; i < N / NP - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                B[(i * LD) + j] = (A[(i + 1) * LD + j] + A[(i - 1) * LD + j] + A[(i * LD) + (j - 1)] + A[(i * LD) + (j + 1)]) * 0.25;
            }
        }

        // Se non è il primo processo, calcolo la prima riga
        if (id != 0) {
            MPI_Wait(&reqs[1], &status);
            for (j = 1; j < N - 1; j++) {
                B[j] = (daprev[j] + A[1 * LD + j] + A[j - 1] + A[j + 1]) * 0.25;
            }
        }

        // Se non è l'ultimo processo, calcolo l'ultima riga
        if (id != NP - 1) {
            MPI_Wait(&reqs[3], &status);
            for (j = 1; j < N - 1; j++) {
                B[((N / NP - 1) * LD) + j] = (danext[j] + A[((N / NP - 1) - 1) * LD + j] + A[((N / NP - 1) * LD) + (j - 1)] + A[((N / NP - 1) * LD) + (j + 1)]) * 0.25;
            }
        }

        // Se non è il primo processo, copio la prima riga
        if (id != 0) {
            for (j = 1; j < N - 1; j++) {
                A[j] = B[j];
            }
        }

        // Copio la matrice interna ad esclusione della prima ed ultima riga e della prima ed ultima colonna
        for (i = 1; i < N / NP - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                A[(i * LD) + j] = B[(i * LD) + j];
            }
        }

        // Se non è l'ultimo processo, copio l'ultima riga
        if (id != NP - 1) {
            for (j = 1; j < N - 1; j++) {
                A[((N / NP - 1) * LD) + j] = B[((N / NP - 1) * LD) + j];
            }
        }
    }

    return;
}



