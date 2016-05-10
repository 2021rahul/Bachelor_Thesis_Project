#ifndef X_MATRIX_H_INCLUDED
#define X_MATRIX_H_INCLUDED

/*******************************************************************
 * The matrix of training vectors is a x_matrix_t structure, because
 * it is not necessarily a dense matrix. Several encodings are used
 * for x_matrix_t's (see below).
 *
 * The main operations used during learning are: 
 * 
 * - x_matrix_dotprod: compute w * x(:, i) 
 * 
 * - x_matrix_addto: w += alpha * x(:, i) 
 */



/*******************************************************************
 * operations on x_matrix_t 
 *
 * handle different encodings of training examples x. 
 * 
 */

#ifndef sparse_index_t 
#define sparse_index_t int
#define sparse_data_t float 
#endif

typedef struct {
  enum {
    JSGD_X_FULL,        /* plain dense matrix */
    JSGD_X_SPARSE,      /* sparse matrix */
    JSGD_X_PQ,          /* PQ-encoded matrix */
    JSGD_X_PQ_SPARSE,   /* Xerox-style sparse PQ (not implemented) */
  } encoding; 

  long n;               /* number of vectors */
  long d;               /* dimension */
  
  /* full matrix */
  float *data;          /* size (d, n) */

  /* pq, 256 centroids */
  int nsq;              /* number of subquantizers */
  float *centroids;     /* size (d / nsq, 256, nsq) */
  unsigned char *codes; /* size (nsq, n) */

  /* sparse matrix */
  sparse_index_t *indices;      /* row indices */
  sparse_index_t *indptr;       /* staring of each column in indices table */
  sparse_data_t *sparse_data;   /* associated data */

  /* sparse PQ */
  int *vector_begin;

} x_matrix_t; 





/*****************************************
 * simple vector operations on dense vectors (SSE accelerated when possible)
 */

/* w := w * fw */
void vec_scale(float *w, long d, float fw);

/* squared norm of vector */
float vec_sqnorm(const float *w, long d);

/* dot product */
float vec_dotprod(const float * __restrict__ xi, const float * __restrict__ w, long d);

/* w += xi * a */
void vec_addto(float * __restrict__ w, float a, const float * __restrict__ xi, long d);


/*****************************************
 * vector-vector operations on x columns
 */


/* return x(:, i) */
/* buffer is of size d that may or may not be used */
const float *x_matrix_get(const x_matrix_t *x, int i, float *buffer);

/* return w * x(:, i) */
double x_matrix_dotprod(const x_matrix_t *x,
                        long i,
                        const float *w);                                 

/* w := w + a * x(:, i)  */
void x_matrix_addto(const x_matrix_t *x,
                    long i,
                    float a,
                    float *w);

/* return x(:, i)' * x(:, j) */
double x_matrix_dotprod_self(const x_matrix_t *x,
                             long i, long j);

/*****************************************
 * x_matrix - dense matrix multiplications (used for classification)
 *
 * x(d, n) 
 * w(d, m)
 * scores(m, n) 
 *
 * scores = w' * x */

void x_matrix_matmul(const x_matrix_t *x,
                     const float *w, 
                     int m, 
                     float *scores);


/* same as x_matrix_matmul, on a subset of i indices */
void x_matrix_matmul_slice(const x_matrix_t *x,
                           int i0, int i1,
                           const float *w, 
                           int m, 
                           float *scores);


/* same, threaded */
void x_matrix_matmul_thread(const x_matrix_t *x,
                            const float *w, 
                            int m, 
                            float *scores);


/*****************************************
 * x_matrix - x_matrix multiplications
 *
 * x(d, n) 
 * y(i1 - i0, i1 - i0) 
 *
 * y = x(:, i0 : i1 - 1)' * x(:, i0 : i1 - 1) */

void x_matrix_matmul_self(const x_matrix_t *x,
                          int i0, int i1, 
                          float *y); 



/* Sparse matrix used in the "use_self_dotprods" implementation of
 * OVR. 
 *
 * The encoding mimics Matlab or scipy.sparse.csc_matrix matrices
 * (and JSGD_X_SPARSE x_matrix_t).
 *
 * See eg. http://www.mathworks.fr/help/techdoc/apiref/mxsetir.html
 */
typedef struct {
  int m, n;   /* nb of rows and columns */
  int *jc;    /* jc[j] points to the the first non-0 cell in column j, in tables ir and pr */
  int *ir;    /* ir[jc[j] : jc[j + 1] - 1] are the non-0 cells in column j */
  float *pr;  /* corresponding values */
} x_matrix_sparse_t; 

/* alloc all in a single chunk, nnz cannot grow... */
void x_matrix_sparse_init(x_matrix_sparse_t *a, int m, int n, int nnz); 

/* free */
void x_matrix_sparse_clear(x_matrix_sparse_t *a); 

/*****************************************
 *
 * compute 
 *
 *    y := x' * w
 *
 * but only for existing entries of y (ie. use ir and jc to compute pr).
 * sizes:
 * 
 *   x(d, n)
 *   w(d, nw)
 */
void x_matrix_matmul_subset(const x_matrix_t *x,
                            x_matrix_sparse_t *y, 
                            const float *ws, 
                            int nw, 
                            int d_step); 
                            
/*****************************************
 *
 * compute 
 * 
 *    w := x * a + w * diag(betas)
 * 
 * sizes:
 * 
 *   x(d, n)
 *   w(d, nw)
 */


void x_matrix_addto_sparse(const x_matrix_t *x,
                           const x_matrix_sparse_t *a,
                           const float *betas,
                           float *ws, 
                           int nw, 
                           int d_step);


#endif

