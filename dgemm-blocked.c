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
#define SECOND_BLOCK_SIZE 96
#endif
#ifndef THIRD_BLOCK_SIZE
#define THIRD_BLOCK_SIZE 48
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

void do_block_4(int lda, double *A, double *B, double *C);

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

static void packing_padding(int lda, int M, int N, double *A, double *A_copy){
  for (int i = 0; i < M; i++)
    
    memcpy(A_copy + i * THIRD_BLOCK_SIZE, A + i * lda, sizeof(double) * N);
    
}
static void do_block_kernel(int lda, int M, int N, int K, double * restrict A, double * restrict B, double * restrict C)
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

  
  for (i = 0; i < M; i += 4)
    for (j = 0; j < N; j += 8)
      {
        //do_test_block_4_unroll(n,padding_a + i * n + k, padding_b + k * n + j, padding_c + i * n + j);
        do_block_48(n,K,padding_a + i * n + k, padding_b + k * n + j, padding_c + i * n + j);
      }

  for (i = 0; i < M; i++)
    for(j = 0; j < N; j++){
    C[i * lda+j] += padding_c [i*n+j];
  }
#else
  double padding_c[n*n * sizeof(double)];

  memset (padding_c, 0, n*n * sizeof(double));
  packing_padding(lda,  M, N, C, padding_c);
  for (int i = 0; i < M; i += REGISTER_BLOCK_SIZE)
    for (int j = 0; j < N; j += REGISTER_BLOCK_SIZE)
      for (int k = 0; k < K; k += REGISTER_BLOCK_SIZE)
      {
        do_block_4_unroll(lda,n, A + i * lda + k, B + k * lda + j, padding_c + i * n + j);
      }
    for (i = 0; i < M; i++)
    for(j = 0; j < N; j++){
    C[i * lda+j] = padding_c [i*n+j];
  }

  
  #endif
}

#ifdef BLOCKS
#include <immintrin.h>
#include <avx2intrin.h>
void do_test_block_4_unroll(int n,double *A, double *B, double *C)
{
  int lda = n;
  register __m256d c0x = _mm256_loadu_pd(C);
  register __m256d c1x = _mm256_loadu_pd(C + lda);
  register __m256d c2x = _mm256_loadu_pd(C + 2 * lda);
  register __m256d c3x = _mm256_loadu_pd(C + 3 * lda);

  register __m256d a0x = _mm256_broadcast_sd(A + 0);
    register __m256d a1x = _mm256_broadcast_sd(A + 0 + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + 0 + 2 * lda);
    register __m256d a3x = _mm256_broadcast_sd(A + 0 + 3 * lda);
    register __m256d b = _mm256_loadu_pd(B + 0 * lda);

    c0x = _mm256_fmadd_pd(a0x, b, c0x);
    c1x = _mm256_fmadd_pd(a1x, b, c1x);
    c2x = _mm256_fmadd_pd(a2x, b, c2x);
    c3x = _mm256_fmadd_pd(a3x, b, c3x);

    a0x = _mm256_broadcast_sd(A + 1);
    a1x = _mm256_broadcast_sd(A + 1 + lda);
    a2x = _mm256_broadcast_sd(A + 1 + 2 * lda);
    a3x = _mm256_broadcast_sd(A + 1 + 3 * lda);
    b = _mm256_loadu_pd(B + 1 * lda);

    c0x = _mm256_fmadd_pd(a0x, b, c0x);
    c1x = _mm256_fmadd_pd(a1x, b, c1x);
    c2x = _mm256_fmadd_pd(a2x, b, c2x);
    c3x = _mm256_fmadd_pd(a3x, b, c3x);

    a0x = _mm256_broadcast_sd(A + 2);
     a1x = _mm256_broadcast_sd(A + 2 + lda);
    a2x = _mm256_broadcast_sd(A + 2 + 2 * lda);
    a3x = _mm256_broadcast_sd(A + 2 + 3 * lda);
    b = _mm256_loadu_pd(B + 2 * lda);

    c0x = _mm256_fmadd_pd(a0x, b, c0x);
    c1x = _mm256_fmadd_pd(a1x, b, c1x);
    c2x = _mm256_fmadd_pd(a2x, b, c2x);
    c3x = _mm256_fmadd_pd(a3x, b, c3x);

    a0x = _mm256_broadcast_sd(A + 3);
    a1x = _mm256_broadcast_sd(A + 3 + lda);
   a2x = _mm256_broadcast_sd(A + 3 + 2 * lda);
    a3x = _mm256_broadcast_sd(A + 3 + 3 * lda);
     b = _mm256_loadu_pd(B + 3 * lda);

    c0x = _mm256_fmadd_pd(a0x, b, c0x);
    c1x = _mm256_fmadd_pd(a1x, b, c1x);
    c2x = _mm256_fmadd_pd(a2x, b, c2x);
    c3x = _mm256_fmadd_pd(a3x, b, c3x);
  
  _mm256_storeu_pd(C, c0x);
  _mm256_storeu_pd(C + lda, c1x);
  _mm256_storeu_pd(C + 2 * lda, c2x);
  _mm256_storeu_pd(C + 3 * lda, c3x);
}
void do_block_48(int n,int K, double *A, double *B, double *C){
  int lda = n;
  register  __m256d c0x = _mm256_loadu_pd(C);
  register __m256d c1x = _mm256_loadu_pd(C + lda);
  register __m256d c2x = _mm256_loadu_pd(C + 2*lda);
  register __m256d c3x = _mm256_loadu_pd(C + 3*lda);

  register __m256d c04x = _mm256_loadu_pd(C+4);
  register __m256d c14x = _mm256_loadu_pd(C + lda + 4);
  register __m256d c24x = _mm256_loadu_pd(C+ 2*lda + 4);
  register __m256d c34x = _mm256_loadu_pd(C + 3*lda + 4);
  
  for (int kk = 0; kk < K; kk++)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3* lda);
    register __m256d b1 = _mm256_loadu_pd(B + kk * lda );
    register __m256d b2 = _mm256_loadu_pd(B + kk * lda + 4 );

    c0x = _mm256_fmadd_pd(a0x, b1, c0x);
    c1x = _mm256_fmadd_pd(a1x, b1, c1x);
    c2x = _mm256_fmadd_pd(a2x, b1, c2x);
    c3x = _mm256_fmadd_pd(a3x, b1, c3x);

    c04x = _mm256_fmadd_pd(a0x, b2, c04x);
    c14x = _mm256_fmadd_pd(a1x, b2, c14x);
    c24x = _mm256_fmadd_pd(a2x, b2, c24x);
    c34x = _mm256_fmadd_pd(a3x, b2, c34x);
  }
  _mm256_storeu_pd(C, c0x);
  _mm256_storeu_pd(C + lda, c1x);
  _mm256_storeu_pd(C + 2*lda , c2x);
  _mm256_storeu_pd(C + 3*lda, c3x);

  _mm256_storeu_pd(C + 4, c04x);
  _mm256_storeu_pd(C + lda + 4, c14x);
  _mm256_storeu_pd(C + 2*lda + 4, c24x);
  _mm256_storeu_pd(C + 3*lda + 4, c34x);

}
void do_block_312(int n,int K, double* restrict A, double* restrict B, double* restrict C){
  int lda = n;
  register __m256d c0x = _mm256_loadu_pd(C);
  register __m256d c1x = _mm256_loadu_pd(C + lda);
  register __m256d c2x = _mm256_loadu_pd(C + 2*lda);

  register __m256d c04x = _mm256_loadu_pd(C+4);
  register __m256d c14x = _mm256_loadu_pd(C + lda + 4);
  register __m256d c24x = _mm256_loadu_pd(C+ 2*lda + 4);

  register __m256d c08x = _mm256_loadu_pd(C+8);
  register __m256d c18x = _mm256_loadu_pd(C + lda + 8);
  register __m256d c28x = _mm256_loadu_pd(C+ 2*lda + 8);
  
  for (int kk = 0; kk < K; kk++)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    
    register __m256d b1 = _mm256_loadu_pd(B + kk * lda );
    register __m256d b2 = _mm256_loadu_pd(B + kk * lda + 4 );
    register __m256d b3 = _mm256_loadu_pd(B + kk * lda + 8 );

    c0x = _mm256_fmadd_pd(a0x, b1, c0x);
    c1x = _mm256_fmadd_pd(a1x, b1, c1x);
    c2x = _mm256_fmadd_pd(a2x, b1, c2x);
    
    c04x = _mm256_fmadd_pd(a0x, b2, c04x);
    c14x = _mm256_fmadd_pd(a1x, b2, c14x);
    c24x = _mm256_fmadd_pd(a2x, b2, c24x);

    c08x = _mm256_fmadd_pd(a0x, b3, c08x);
    c18x = _mm256_fmadd_pd(a1x, b3, c18x);
    c28x = _mm256_fmadd_pd(a2x, b3, c28x);
  }
  _mm256_storeu_pd(C, c0x);
  _mm256_storeu_pd(C + lda, c1x);
  _mm256_storeu_pd(C + 2*lda , c2x);

  _mm256_storeu_pd(C + 4, c04x);
  _mm256_storeu_pd(C + lda + 4, c14x);
  _mm256_storeu_pd(C + 2*lda + 4, c24x);

  _mm256_storeu_pd(C + 8, c08x);
  _mm256_storeu_pd(C + lda + 8, c18x);
  _mm256_storeu_pd(C + 2*lda + 8, c28x);


}

void do_block_28(int n, double *A, double *B, double *C){
  int lda = n;
  register __m256d c0x = _mm256_loadu_pd(C);
  register __m256d c1x = _mm256_loadu_pd(C + lda);
  register __m256d c04x = _mm256_loadu_pd(C+4);
  register __m256d c14x = _mm256_loadu_pd(C + lda + 4);
  
  for (int kk = 0; kk < 4; kk++)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d b1 = _mm256_loadu_pd(B + kk * lda );
    register __m256d b2 = _mm256_loadu_pd(B + kk * lda + 4 );

    c0x = _mm256_fmadd_pd(a0x, b1, c0x);
    c1x = _mm256_fmadd_pd(a1x, b1, c1x);
    c04x = _mm256_fmadd_pd(a0x, b2, c04x);
    c14x = _mm256_fmadd_pd(a1x, b2, c14x);
  }
  _mm256_storeu_pd(C, c0x);
  _mm256_storeu_pd(C + lda, c1x);
  _mm256_storeu_pd(C + 4, c04x);
  _mm256_storeu_pd(C + lda + 4, c14x);

}
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
    register __m256d b = _mm256_loadu_pd(B + kk * lda );

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
void do_block_4(int lda, double *A, double *B, double *C)
{
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
#endif
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += FIRST_BLOCK_SIZE){
    int M = min(FIRST_BLOCK_SIZE, lda - i);
    /* For each block-column of B */
    for (int j = 0; j < lda; j += FIRST_BLOCK_SIZE){
      int N = min(FIRST_BLOCK_SIZE, lda - j);
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += FIRST_BLOCK_SIZE)
      {
        int K = min(FIRST_BLOCK_SIZE, lda - k);

#ifdef BLOCKS
        //second level blocking
        for (int ii = i; ii < i + M; ii += SECOND_BLOCK_SIZE){
          int MM = min(SECOND_BLOCK_SIZE, i + M - ii);
          for (int jj = j; jj < j + N; jj += SECOND_BLOCK_SIZE){
            int NN = min(SECOND_BLOCK_SIZE, j + N - jj);
            for (int kk = k; kk < k + K; kk += SECOND_BLOCK_SIZE)

            {
              /* Correct block dimensions if block "goes off edge of" the matrix */

              
              
              int KK = min(SECOND_BLOCK_SIZE, k + K - kk);

              //third level blocking
              for (int iii = ii; iii < ii + MM; iii += THIRD_BLOCK_SIZE){
                for (int jjj = jj; jjj < jj + NN; jjj += THIRD_BLOCK_SIZE){
                  for (int kkk = kk; kkk < kk + KK; kkk += THIRD_BLOCK_SIZE)
                  {
                    int MMM = min(THIRD_BLOCK_SIZE, ii + MM - iii);
                    int NNN = min(THIRD_BLOCK_SIZE, jj + NN - jjj);
                    int KKK = min(THIRD_BLOCK_SIZE, kk + KK - kkk);
#endif
                    /* Perform individual block dgemm */

#ifdef BLOCKS
#ifdef PADDING

do_block_kernel(lda, MMM, NNN, KKK, A + iii * lda + kkk, B + kkk * lda + jjj, C + iii * lda + jjj);
#endif
#else
        do_block(lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
#ifdef BLOCKS
                  }
            }
        
#endif
      }
}
}
}
  }
}
  }
}