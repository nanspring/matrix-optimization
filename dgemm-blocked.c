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
#define FIRST_BLOCK_SIZE 576
#endif
#ifndef SECOND_BLOCK_SIZE
#define SECOND_BLOCK_SIZE 288
#endif
#ifndef THIRD_BLOCK_SIZE
#define THIRD_BLOCK_SIZE 144
#endif

double padding_a[THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE] __attribute__((aligned(256)));
double padding_b[THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE] __attribute__((aligned(256)));
double padding_c[THIRD_BLOCK_SIZE * THIRD_BLOCK_SIZE] __attribute__((aligned(256)));


#define min(a, b) (((a) < (b)) ? (a) : (b))

void do_block_312(int n, int K, double *restrict A, double *restrict B, double *restrict C);

static void packing_padding(int lda, int M, int N, double *A, double *A_copy)
{
  for (int i = 0; i < M; i++)

    memcpy(A_copy + i * THIRD_BLOCK_SIZE, A + i * lda, sizeof(double) * N);
}
static void do_block_kernel(int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
  int n = THIRD_BLOCK_SIZE;
  int i, j;
  memset(padding_a, 0, n * n * sizeof(double));
  memset(padding_b, 0, n * n * sizeof(double));
  memset(padding_c, 0, n * n * sizeof(double));

  packing_padding(lda, M, K, A, padding_a);
  packing_padding(lda, K, N, B, padding_b);

  for (i = 0; i < M; i += 3)
    for (j = 0; j < N; j += 12)
    {
      do_block_312(n, K, padding_a + i * n, padding_b + j, padding_c + i * n + j);
    }

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
    {
      C[i * lda + j] += padding_c[i * n + j];
    }
}

#include <immintrin.h>
#include <avx2intrin.h>

void do_block_312(int n, int K, double *restrict A, double *restrict B, double *restrict C)
{
  int lda = n;
  register __m256d c0x = _mm256_loadu_pd(C);
  register __m256d c1x = _mm256_loadu_pd(C + lda);
  register __m256d c2x = _mm256_loadu_pd(C + 2 * lda);

  register __m256d c04x = _mm256_loadu_pd(C + 4);
  register __m256d c14x = _mm256_loadu_pd(C + lda + 4);
  register __m256d c24x = _mm256_loadu_pd(C + 2 * lda + 4);

  register __m256d c08x = _mm256_loadu_pd(C + 8);
  register __m256d c18x = _mm256_loadu_pd(C + lda + 8);
  register __m256d c28x = _mm256_loadu_pd(C + 2 * lda + 8);

  for (int kk = 0; kk < K; kk++)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2 * lda);

    register __m256d b1 = _mm256_loadu_pd(B + kk * lda);
    register __m256d b2 = _mm256_loadu_pd(B + kk * lda + 4);
    register __m256d b3 = _mm256_loadu_pd(B + kk * lda + 8);

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
  _mm256_storeu_pd(C + 2 * lda, c2x);

  _mm256_storeu_pd(C + 4, c04x);
  _mm256_storeu_pd(C + lda + 4, c14x);
  _mm256_storeu_pd(C + 2 * lda + 4, c24x);

  _mm256_storeu_pd(C + 8, c08x);
  _mm256_storeu_pd(C + lda + 8, c18x);
  _mm256_storeu_pd(C + 2 * lda + 8, c28x);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += FIRST_BLOCK_SIZE)
  {
    int M = min(FIRST_BLOCK_SIZE, lda - i);
    /* For each block-column of B */
    for (int j = 0; j < lda; j += FIRST_BLOCK_SIZE)
    {
      int N = min(FIRST_BLOCK_SIZE, lda - j);
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += FIRST_BLOCK_SIZE)
      {
        int K = min(FIRST_BLOCK_SIZE, lda - k);
        //second level blocking
        for (int ii = i; ii < i + M; ii += SECOND_BLOCK_SIZE)
        {
          int MM = min(SECOND_BLOCK_SIZE, i + M - ii);
          for (int jj = j; jj < j + N; jj += SECOND_BLOCK_SIZE)
          {
            int NN = min(SECOND_BLOCK_SIZE, j + N - jj);
            for (int kk = k; kk < k + K; kk += SECOND_BLOCK_SIZE)

            {
              /* Correct block dimensions if block "goes off edge of" the matrix */

              int KK = min(SECOND_BLOCK_SIZE, k + K - kk);

              //third level blocking
              for (int iii = ii; iii < ii + MM; iii += THIRD_BLOCK_SIZE)
              {
                for (int jjj = jj; jjj < jj + NN; jjj += THIRD_BLOCK_SIZE)
                {
                  for (int kkk = kk; kkk < kk + KK; kkk += THIRD_BLOCK_SIZE)
                  {
                    int MMM = min(THIRD_BLOCK_SIZE, ii + MM - iii);
                    int NNN = min(THIRD_BLOCK_SIZE, jj + NN - jjj);
                    int KKK = min(THIRD_BLOCK_SIZE, kk + KK - kkk);
                    /* Perform individual block dgemm */

                    do_block_kernel(lda, MMM, NNN, KKK, A + iii * lda + kkk, B + kkk * lda + jjj, C + iii * lda + jjj);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
} 
