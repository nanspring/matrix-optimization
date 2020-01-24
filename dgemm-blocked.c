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
#ifdef PADDING
double padding_a[THIRD_BLOCK_SIZE*THIRD_BLOCK_SIZE * sizeof(double)];
double padding_b[THIRD_BLOCK_SIZE*THIRD_BLOCK_SIZE * sizeof(double)];
double padding_c[THIRD_BLOCK_SIZE*THIRD_BLOCK_SIZE * sizeof(double)];
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

static void do_block_kernel(int lda, int M, int N, int K, double *A, double *B, double *C)
{
#ifdef PADDING
if (M != THIRD_BLOCK_SIZE || N != THIRD_BLOCK_SIZE || K != THIRD_BLOCK_SIZE){
  int i,j,k;
  memset (padding_a, 0, THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE * sizeof(double));
  memset (padding_b, 0, THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE * sizeof(double));
  memset (padding_c, 0, THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE * sizeof(double));

  // for (i = 0; i < M; i++){
  //   memcpy(padding_a + i * lda, A + i * lda, sizeof(double) * K);
  // }
  // for (i = 0; i < K; i++){
  //   memcpy(padding_b + i * lda, B + i * lda, sizeof(double) * N);
  // }
  // for (i = 0; i < M; i++){
  //   memcpy(padding_c + i * lda, C + i * lda, sizeof(double) * N);
  // }
  // for (i = 0; i < M; i += REGISTER_BLOCK_SIZE)
  //   for (j = 0; j < N; j += REGISTER_BLOCK_SIZE)
  //     for (k = 0; k < K; k += REGISTER_BLOCK_SIZE)
  //     {
  //       do_block_4(lda, padding_a + i * lda + k, padding_b + k * lda + j, padding_c + i * lda + j);
  //     }
  // for (i = 0; i < M; i++){
  //   memcpy(C + i * lda, padding_c + i * lda, sizeof(double) * N);
  // }
}
#endif
  for (int i = 0; i < M; i += REGISTER_BLOCK_SIZE)
    for (int j = 0; j < N; j += REGISTER_BLOCK_SIZE)
      for (int k = 0; k < K; k += REGISTER_BLOCK_SIZE)
      {
        do_block_4(lda, A + i * lda + k, B + k * lda + j, C + i * lda + j);
      }
}

#ifdef BLOCKS
#include <immintrin.h>
#include <avx2intrin.h>
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