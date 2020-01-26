/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <string.h>
const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef REGISTER_BLOCK_SIZE
#define REGISTER_BLOCK_SIZE 4
#endif
#ifndef FIRST_BLOCK_SIZE
#define FIRST_BLOCK_SIZE 384
#endif
#ifndef SECOND_BLOCK_SIZE
#define SECOND_BLOCK_SIZE 64
#endif
#ifndef THIRD_BLOCK_SIZE
#define THIRD_BLOCK_SIZE 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

void do_block_4(int lda,int cn, double *A, double *B, double *C);

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      register double cij = C[i * lda + j];
      for (int k = 0; k < K; ++k)
        cij += A[i * lda + k] * B[k * lda + j];
      C[i * lda + j] = cij;
    }
}
static void printMatrix(int N, double *A){
  int column;
 
      for(column=0; column<8; column++)
          {
          printf("%f ", A[N-8+column]);
          }
          printf("%s", "|");
      printf("\n");
      printf("-----------------------");
      printf("\n");
  
}
static void packing_padding(int lda, int M, int N, double *A, double *A_copy){
  for (int i = 0; i < M; i++)
    //for(int j = 0; j < N; j++){
      //A_copy[i * THIRD_BLOCK_SIZE+j] = A[i * lda+j];
    memcpy(A_copy + i * THIRD_BLOCK_SIZE, A + i * lda, sizeof(double) * N);
    
  //}
  //printf("%s\n","out");
}
static void do_block_kernel(int lda, int M, int N, int K, double *A, double *B, double *C)
{
   int n = THIRD_BLOCK_SIZE;
    int i,j,k;
   
#ifdef PADDING
  
  double padding_a[n*n * sizeof(double)];
  double padding_b[n*n * sizeof(double)];
  double padding_c[n*n * sizeof(double)];
  memset (padding_a, 0, n*n * sizeof(double));
  memset (padding_b, 0, n*n * sizeof(double));
  memset (padding_c, 0, n*n * sizeof(double));
  
  packing_padding(lda, M, K, A,padding_a);
  packing_padding(lda, K, N, B,padding_b);
  packing_padding(lda,  M, N, C,padding_c);

  // printMatrix( K,A);
  // printMatrix(THIRD_BLOCK_SIZE,padding_a);
  // printf("++++++++++++++++++++\n");
  for (i = 0; i < M; i += REGISTER_BLOCK_SIZE)
    for (j = 0; j < N; j += REGISTER_BLOCK_SIZE)
      for (k = 0; k < K; k += REGISTER_BLOCK_SIZE)
      {
        do_test_block_4(n,padding_a + i * n + k, padding_b + k * n + j, padding_c + i * n + j);
      }

  for (i = 0; i < M; i++)
    for(j = 0; j < N; j++){
    C[i * lda+j] = padding_c [i*n+j];
  }

  

#else
  double padding_c[n*n * sizeof(double)];
  //int ldc = 4;
  memset (padding_c, 0, n*n * sizeof(double));
  packing_padding(lda,  M, N, C, padding_c);
  for (int i = 0; i < M; i += REGISTER_BLOCK_SIZE)
    for (int j = 0; j < N; j += REGISTER_BLOCK_SIZE)
      for (int k = 0; k < K; k += REGISTER_BLOCK_SIZE)
      {
        do_block_4(lda,n, A + i * lda + k, B + k * lda + j, padding_c + i * n + j);
      }
    for (i = 0; i < M; i++)
    for(j = 0; j < N; j++){
    C[i * lda+j] = padding_c [i*n+j];
  }
  // printMatrix(n,padding_c);
  // for (int i = 0; i < M; i += REGISTER_BLOCK_SIZE)
  //   for (int j = 0; j < N; j += REGISTER_BLOCK_SIZE)
  //     for (int k = 0; k < K; k += REGISTER_BLOCK_SIZE)
  //     {
  //       do_block_4(lda,lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
  //     }
  //  printMatrix(N,C);
  // printf("+++++++++++++++++\n");
  //     for (i = 0; i < M; i++)
  //   for(j = 0; j < N; j++){
  //   C[i * lda+j] += padding_c [i*n+j];
  // }
  #endif
}

#ifdef BLOCKS
#include <immintrin.h>
#include <avx2intrin.h>
void do_test_block_4(int n,double *A, double *B, double *C)
{
  int lda = n;
  register __m256d c0x = _mm256_loadu_pd(C);
  register __m256d c1x = _mm256_loadu_pd(C + lda);
  register __m256d c2x = _mm256_loadu_pd(C + 2 * lda);
  register __m256d c3x = _mm256_loadu_pd(C + 3 * lda);
  for (int kk = 0; kk < 4; kk++)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2 * lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3 * lda);
    register __m256d b = _mm256_loadu_pd(B + kk * lda);

    c0x = _mm256_fmadd_pd(a0x, b, c0x);
    c1x = _mm256_fmadd_pd(a1x, b, c1x);
    c2x = _mm256_fmadd_pd(a2x, b, c2x);
    c3x = _mm256_fmadd_pd(a3x, b, c3x);
  }
  _mm256_storeu_pd(C, c0x);
  _mm256_storeu_pd(C + lda, c1x);
  _mm256_storeu_pd(C + 2 * lda, c2x);
  _mm256_storeu_pd(C + 3 * lda, c3x);
}
/* C[4*4] = A[4*4] * B[4*4]
*/
void do_block_4(int lda,int cn, double *A, double *B, double *C)
{
  register __m256d c0x = _mm256_loadu_pd(C);
  register __m256d c1x = _mm256_loadu_pd(C + cn);
  register __m256d c2x = _mm256_loadu_pd(C + 2 * cn);
  register __m256d c3x = _mm256_loadu_pd(C + 3 * cn);
  for (int kk = 0; kk < 4; kk++)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2 * lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3 * lda);
    register __m256d b = _mm256_loadu_pd(B + kk * lda);

    c0x = _mm256_fmadd_pd(a0x, b, c0x);
    c1x = _mm256_fmadd_pd(a1x, b, c1x);
    c2x = _mm256_fmadd_pd(a2x, b, c2x);
    c3x = _mm256_fmadd_pd(a3x, b, c3x);
  }
  _mm256_storeu_pd(C, c0x);
  _mm256_storeu_pd(C + cn, c1x);
  _mm256_storeu_pd(C + 2 * cn, c2x);
  _mm256_storeu_pd(C + 3 * cn, c3x);
}
#endif
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += FIRST_BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += FIRST_BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += FIRST_BLOCK_SIZE)
      {
        int M = min(FIRST_BLOCK_SIZE, lda - i);
        int N = min(FIRST_BLOCK_SIZE, lda - j);
        int K = min(FIRST_BLOCK_SIZE, lda - k);

#ifdef BLOCKS
        //second level blocking
        for (int ii = i; ii < i + M; ii += SECOND_BLOCK_SIZE)
          for (int jj = j; jj < j + N; jj += SECOND_BLOCK_SIZE)
            for (int kk = k; kk < k + K; kk += SECOND_BLOCK_SIZE)

            {
              /* Correct block dimensions if block "goes off edge of" the matrix */

              int MM = min(SECOND_BLOCK_SIZE, i + M - ii);
              int NN = min(SECOND_BLOCK_SIZE, j + N - jj);
              int KK = min(SECOND_BLOCK_SIZE, k + K - kk);

              //third level blocking
              for (int iii = ii; iii < ii + MM; iii += THIRD_BLOCK_SIZE)
                for (int jjj = jj; jjj < jj + NN; jjj += THIRD_BLOCK_SIZE)
                  for (int kkk = kk; kkk < kk + KK; kkk += THIRD_BLOCK_SIZE)
                  {
                    int MMM = min(THIRD_BLOCK_SIZE, ii + MM - iii);
                    int NNN = min(THIRD_BLOCK_SIZE, jj + NN - jjj);
                    int KKK = min(THIRD_BLOCK_SIZE, kk + KK - kkk);
#endif

                    /* Perform individual block dgemm */

#ifdef BLOCKS
#ifdef PADDING
// if (MMM == THIRD_BLOCK_SIZE && NNN == THIRD_BLOCK_SIZE && KKK == THIRD_BLOCK_SIZE)
//         {
//           do_block_kernel(lda, MMM, NNN, KKK, A + iii * lda + kkk, B + kkk * lda + jjj, C + iii * lda + jjj);
//         }
//         else
//         {
//           do_block(lda, MMM, NNN, KKK, A + iii * lda + kkk, B + kkk * lda + jjj, C + iii * lda + jjj);
//         }
do_block_kernel(lda, MMM, NNN, KKK, A + iii * lda + kkk, B + kkk * lda + jjj, C + iii * lda + jjj);
#else
        if (MMM == THIRD_BLOCK_SIZE && NNN == THIRD_BLOCK_SIZE && KKK == THIRD_BLOCK_SIZE)
        {
          do_block_kernel(lda, MMM, NNN, KKK, A + iii * lda + kkk, B + kkk * lda + jjj, C + iii * lda + jjj);
        }
        else
        {
          do_block(lda, MMM, NNN, KKK, A + iii * lda + kkk, B + kkk * lda + jjj, C + iii * lda + jjj);
        }
#endif
#else
        do_block(lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
#ifdef BLOCKS
                  }
            }
           // printMatrix(THIRD_BLOCK_SIZE,padding_a);
#endif
      }
}