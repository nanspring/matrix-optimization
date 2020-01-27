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
#ifdef PADDING
double padding_a[THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE] __attribute__((aligned(256)));
double padding_b[THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE] __attribute__((aligned(256)));
double padding_c[THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE] __attribute__((aligned(256)));
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

// void do_block_register(int lda, double *A, double *B, double *C);
void do_block_48(int n, int K, double *A, double *B, double *C);

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
      double cij = C[i * lda + j];
      for (int k = 0; k < K; ++k)
        cij += A[i * lda + k] * B[k * lda + j];
      C[i * lda + j] = cij;
    }
}

static void do_block_kernel(int lda, int M, int N, int K, double *A, double *B, double *C)
{
#ifdef PADDING
  int i, j, k;
  memset(padding_a, 0, THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE * sizeof(double));
  memset(padding_b, 0, THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE * sizeof(double));
  memset(padding_c, 0, THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE * sizeof(double));

  for (i = 0; i < M; i++)
  {
    memcpy(padding_a + i * THIRD_BLOCK_SIZE, A + i * lda, sizeof(double) * K);
  }
  for (i = 0; i < K; i++)
  {
    memcpy(padding_b + i * THIRD_BLOCK_SIZE, B + i * lda, sizeof(double) * N);
  }

  for (i = 0; i < M; i += 4)
    for (j = 0; j < N; j += 8)
      {
        do_block_48(THIRD_BLOCK_SIZE, K, padding_a + i * THIRD_BLOCK_SIZE + k, padding_b + k * THIRD_BLOCK_SIZE + j, padding_c + i * THIRD_BLOCK_SIZE + j);
      }
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
    {
      C[i * lda + j] += padding_c[i * THIRD_BLOCK_SIZE + j];
    }
#else
  for (int i = 0; i < M; i += REGISTER_BLOCK_SIZE)
    for (int j = 0; j < N; j += 8)
      for (int k = 0; k < K; k += REGISTER_BLOCK_SIZE)
      {
        do_block_48(lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
      }
#endif
}

#ifdef BLOCKS
#include <immintrin.h>
#include <avx2intrin.h>
/* C[4*8] = A[4*4] * B[4*8]
*/
void do_block_48(int lda,int K, double *A, double *B, double *C){
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
    for (int k = 0; k < lda; k += FIRST_BLOCK_SIZE)
      for (int j = 0; j < lda; j += FIRST_BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      {
        int M = min(FIRST_BLOCK_SIZE, lda - i);
        int N = min(FIRST_BLOCK_SIZE, lda - j);
        int K = min(FIRST_BLOCK_SIZE, lda - k);

#ifdef BLOCKS
        //second level blocking
        for (int ii = i; ii < i + M; ii += SECOND_BLOCK_SIZE)
          for (int kk = k; kk < k + K; kk += SECOND_BLOCK_SIZE)
            for (int jj = j; jj < j + N; jj += SECOND_BLOCK_SIZE)

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
#endif
      }
}