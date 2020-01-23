/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char* dgemm_desc = "Simple blocked dgemm.";


#include <immintrin.h>
#include <avx2intrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 416
#define SECOND_BLOCK_SIZE 64
#define THIRD_BLOCK_SIZE 32
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
	cij += A[i*lda+k] * B[j*lda+k];
#else
	cij += A[i*lda+k] * B[k*lda+j];
#endif
      C[i*lda+j] = cij;
      //printf("%d\n",cij);
    }
}
static void do_reg4x4_block (int lda, double* A, double* B, double* C){
  int kk;
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*lda);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+3*lda);

    
  for(kk = 0; kk < 4; kk++){
    register __m256d a0x = _mm256_broadcast_sd(A+kk+0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A+kk+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+kk+3*lda);
    register __m256d b = _mm256_loadu_pd(B+kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x,b,c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x,b,c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x,b,c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x,b,c30_c31_c32_c33);
  }
  _mm256_storeu_pd(C,c00_c01_c02_c03);
  _mm256_storeu_pd(C+lda,c10_c11_c12_c13);
  _mm256_storeu_pd(C+2*lda,c20_c21_c22_c23);
  _mm256_storeu_pd(C+3*lda,c30_c31_c32_c33);
}
static void do_reg4x4_block_unrolling (int lda, double* A, double* B, double* C)
{
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*lda);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+3*lda);

    register __m256d a0x = _mm256_broadcast_sd(A+0+0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A+0+1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A+0+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+0+3*lda);
    register __m256d b = _mm256_loadu_pd(B+0*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x,b,c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x,b,c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x,b,c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x,b,c30_c31_c32_c33);

    a0x = _mm256_broadcast_sd(A+1+0*lda);
    a1x = _mm256_broadcast_sd(A+1+1*lda);
    a2x = _mm256_broadcast_sd(A+1+2*lda);
    a3x = _mm256_broadcast_sd(A+1+3*lda);
    b = _mm256_loadu_pd(B+1*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x,b,c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x,b,c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x,b,c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x,b,c30_c31_c32_c33);

    a0x = _mm256_broadcast_sd(A+2+0*lda);
    a1x = _mm256_broadcast_sd(A+2+1*lda);
     a2x = _mm256_broadcast_sd(A+2+2*lda);
    a3x = _mm256_broadcast_sd(A+2+3*lda);
    b = _mm256_loadu_pd(B+2*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x,b,c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x,b,c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x,b,c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x,b,c30_c31_c32_c33);

    a0x = _mm256_broadcast_sd(A+3+0*lda);
     a1x = _mm256_broadcast_sd(A+3+1*lda);
   a2x = _mm256_broadcast_sd(A+3+2*lda);
     a3x = _mm256_broadcast_sd(A+3+3*lda);
     b = _mm256_loadu_pd(B+3*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x,b,c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x,b,c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x,b,c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x,b,c30_c31_c32_c33);

    
  
  //printf("%d %d %d %d\n",c00_c01_c02_c03,c10_c11_c12_c13,c20_c21_c22_c23,c30_c31_c32_c33);
  _mm256_storeu_pd(C,c00_c01_c02_c03);
  _mm256_storeu_pd(C+lda,c10_c11_c12_c13);
  _mm256_storeu_pd(C+2*lda,c20_c21_c22_c23);
  _mm256_storeu_pd(C+3*lda,c30_c31_c32_c33);
  //printf("%f %f %f %f\n",*C,*C+lda,*C+2*lda,*C+3*lda);
    
}
static void inner_kernel (int lda, int M, int N, int K, double* A, double* B, double* C){
    int i,j,k;
    for( i = 0; i < M; i +=4)
    for(j = 0; j < N; j += 4)
    for(k = 0; k < K; k += 4)
    
    {
      
        do_reg4x4_block_unrolling(lda,A + i*lda + k, B + k*lda + j, C + i*lda + j);
      
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
    }
#endif
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int k = 0; k < lda; k += BLOCK_SIZE)
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      {
	int M = min(BLOCK_SIZE,lda-i);
	int N = min(BLOCK_SIZE,lda-j);
	int K = min(BLOCK_SIZE,lda-k);

#ifdef BLOCKS
  //second level blocking
	for(int ii = i; ii < i+M; ii += SECOND_BLOCK_SIZE)
    for(int kk = k; kk < k+K; kk += SECOND_BLOCK_SIZE)
	  for(int jj = j; jj < j+N; jj += SECOND_BLOCK_SIZE)
	    

	{
	/* Correct block dimensions if block "goes off edge of" the matrix */

	int MM = min (SECOND_BLOCK_SIZE,i+ M-ii);
	int NN = min (SECOND_BLOCK_SIZE,j+ N-jj);
	int KK = min (SECOND_BLOCK_SIZE,k+ K-kk);

  //third level blocking
  for(int iii = ii; iii < ii+MM; iii += THIRD_BLOCK_SIZE)
    for(int jjj = jj; jjj < jj+NN; jjj += THIRD_BLOCK_SIZE)
      for(int kkk = kk; kkk < kk+KK; kkk += THIRD_BLOCK_SIZE){
        int MMM = min (THIRD_BLOCK_SIZE,ii+ MM-iii);
	      int NNN = min (THIRD_BLOCK_SIZE,jj+ NN-jjj);
	      int KKK = min (THIRD_BLOCK_SIZE,kk+ KK-kkk);
#endif


	/* Perform individual block dgemm */
#ifdef TRANSPOSE
	#ifdef BLOCKS
	do_block(lda,MMM,NNN,KKK,A+iii*lda+kkk,B+jjj*lda+kkk,C+iii*lda+jjj);
	#else
	do_block(lda, M, N, K, A + i*lda + k, B + j*lda + k, C + i*lda + j);
	#endif
#else
	#ifdef BLOCKS
  if(MMM==32 && NNN == 32 && KKK == 32 ){
    inner_kernel(lda,MMM,NNN,KKK,A+iii*lda+kkk, B+ kkk*lda+jjj,C+iii*lda+jjj);
    //do_reg4x4_block (lda,A + i*lda + k, B + k*lda + j, C + i*lda + j);
  }else{
    do_block(lda, MMM, NNN, KKK, A + iii*lda + kkk, B + kkk*lda + jjj, C + iii*lda + jjj);
  }
	
	#else
	do_block(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
	#endif
#endif
      	#ifdef BLOCKS
	  }
  }
	#endif
  }
#if TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
    }
#endif
}
