/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 37
#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE 4
#endif
// #define BLOCK_SIZE 719
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
{
#ifdef REGISTERTILE
  int block_edge_m = M / L1_BLOCK_SIZE * L1_BLOCK_SIZE;
  int block_edge_n = N / L1_BLOCK_SIZE * L1_BLOCK_SIZE;
  int block_edge_k = K / L1_BLOCK_SIZE * L1_BLOCK_SIZE;
  /* compute blocks that can be evenly divided */
  for (int i = 0; i < block_edge_m; i += L1_BLOCK_SIZE)
  {
    for (int j = 0; j < block_edge_n; j += L1_BLOCK_SIZE)
    {
      for (int k = 0; k < block_edge_k; k += L1_BLOCK_SIZE)
      {
        do_block_4(lda, A + i * lda + j, B + k * lda + j, C + i * lda + j);
      }
    }
  }
  /* compute the part at the edge that can not fit into blocks */
  for (int i = block_edge_m; i < M; i++)
  {
    for (int j = block_edge_n; j < N; j++)
    {
      double cij = C[i * lda + j];
      for (int k = block_edge_k; k < K; k++)
      {
        cij += A[i * lda + k] * B[k * lda + j];
      }
      C[i * lda + j] = cij;
    }
  }
#else
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i * lda + j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * lda + j];
#endif
      C[i * lda + j] = cij;
    }
#endif
}

#ifdef REGISTERTILE
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
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i + 1; j < lda; ++j)
    {
      double t = B[i * lda + j];
      B[i * lda + j] = B[j * lda + i];
      B[j * lda + i] = t;
    }
#endif
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min(BLOCK_SIZE, lda - i);
        int N = min(BLOCK_SIZE, lda - j);
        int K = min(BLOCK_SIZE, lda - k);

        /* Perform individual block dgemm */
#ifdef TRANSPOSE
        do_block(lda, M, N, K, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_block(lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
      }
#if TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i + 1; j < lda; ++j)
    {
      double t = B[i * lda + j];
      B[i * lda + j] = B[j * lda + i];
      B[j * lda + i] = t;
    }
#endif
}