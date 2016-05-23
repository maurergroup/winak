//
// This file is Copyright Daniel Strobusch.
//
// C wrapper to SparseSuiteQR library et al. for Python

// We pass in the sparse matrix data in a COO sparse matrix format. Cholmod
// refers to this as a "cholmod_triplet" format. This is then converted to its
// "cholmod_sparse" format, which is a CSC matrix.

#include <stdio.h>
#include <stdlib.h>
#include "suitesparse/SuiteSparseQR_C.h"
void qr_solve(double const *A_data, long const *A_row, long const *A_col, size_t A_nnz, \
        size_t A_m, size_t A_n, double const *b_data, double *x_data) {
    // Solves the matrix equation Ax=b where A is a sparse matrix and x and b
    // are dense column vectors. A and b are inputs, x is solved for in the
    // least squares sense using a rank-revealing QR factorization.
    //
    // Inputs
    //
    // A_data, A_row, A_col: the COO data
    // A_nnz: number of non-zero entries, ie the length of the arrays A_data, etc
    // A_m: number of rows in A
    // A_n: number of cols in A
    // b_data: the data in b. It is A_m entries long.
    //
    // Outputs
    //
    // x_data: the data in x. It is A_n entries long
    //
    // MAKE SURE x_data is allocated to the right size before calling this function
    //
    cholmod_common Common, *cc;
    cholmod_sparse *A_csc;
    cholmod_triplet *A_coo;
    cholmod_dense *b, *x;
    size_t k;
    // Helper pointers
    long *Ai, *Aj;
    double *Ax, *bx, *xx;

    /* start CHOLMOD */
    cc = &Common ;
    cholmod_l_start (cc) ;

    // Create A, first as a COO matrix, then convert to CSC
    A_coo = cholmod_l_allocate_triplet(A_m, A_n, A_nnz, 0, CHOLMOD_REAL, cc);
    if (A_coo == NULL) {
        fprintf(stderr, "ERROR: cannot allocate triplet");
        return;
    }
    // Copy in data
    Ai = A_coo->i;
    Aj = A_coo->j;
    Ax = A_coo->x;
    for (k=0; k<A_nnz; k++) {
        Ai[k] = A_row[k];
        Aj[k] = A_col[k];
        Ax[k] = A_data[k];
    }
    A_coo->nnz = A_nnz;
    // Make sure the matrix is valid
    if (cholmod_l_check_triplet(A_coo, cc) != 1) {
        fprintf(stderr, "ERROR: triplet matrix is not valid");
        return;
    }
    // Convert to CSC
    A_csc = cholmod_l_triplet_to_sparse(A_coo, A_nnz, cc);

    // Create b as a dense matrix
    b = cholmod_l_allocate_dense(A_m, 1, A_m, CHOLMOD_REAL, cc);
    bx = b->x;
    for (k=0; k<A_m; k++) {
        bx[k] = b_data[k];
    }
    // Make sure the matrix is valid
    if (cholmod_l_check_dense(b, cc) != 1) {
        fprintf(stderr, "ERROR: b vector is not valid");
        return;
    }

    // Solve for x
    x = SuiteSparseQR_C_backslash_default(A_csc, b, cc);

    // Return values of x
    xx = x->x;
    for (k=0; k<A_n; k++) {
        x_data[k] = xx[k];
    }

    /* free everything and finish CHOLMOD */
    cholmod_l_free_triplet(&A_coo, cc);
    cholmod_l_free_sparse(&A_csc, cc);
    cholmod_l_free_dense(&x, cc);
    cholmod_l_free_dense(&b, cc);
    cholmod_l_finish(cc);
    return;
}

void qr_getqr(double const *A_data, long const *A_row, long const *A_col, size_t A_nnz, \
        size_t A_m, size_t A_n, double const *b_data, double *x_data) {
    // Calculates and returns Q and R for the matrix A given as sparse matrix
    //
    // Inputs
    //
    // A_data, A_row, A_col: the COO data
    // A_nnz: number of non-zero entries, ie the length of the arrays A_data, etc
    // A_m: number of rows in A
    // A_n: number of cols in A
    // rank: rank of matrix A, e = max(min(m,econ),rank(A))
    // 
    // Outputs
    //
    // Q_data, Q_indices, Q_indptr: csc format data of Q
    // R_data, R_indices, R_indptr: csc format data of R
    //
    // MAKE SURE x_data is allocated to the right size before calling this function
    //
    cholmod_common Common, *cc;
    cholmod_sparse *A_csc;
    cholmod_triplet *A_coo;
    cholmod_dense *b, *x;
    size_t k;
    // Helper pointers
    long *Ai, *Aj;
    double *Ax, *bx, *xx;

    /* start CHOLMOD */
    cc = &Common ;
    cholmod_l_start (cc) ;

    // Create A, first as a COO matrix, then convert to CSC
    A_coo = cholmod_l_allocate_triplet(A_m, A_n, A_nnz, 0, CHOLMOD_REAL, cc);
    if (A_coo == NULL) {
        fprintf(stderr, "ERROR: cannot allocate triplet");
        return;
    }
    // Copy in data
    Ai = A_coo->i;
    Aj = A_coo->j;
    Ax = A_coo->x;
    for (k=0; k<A_nnz; k++) {
        Ai[k] = A_row[k];
        Aj[k] = A_col[k];
        Ax[k] = A_data[k];
    }
    A_coo->nnz = A_nnz;
    // Make sure the matrix is valid
    if (cholmod_l_check_triplet(A_coo, cc) != 1) {
        fprintf(stderr, "ERROR: triplet matrix is not valid");
        return;
    }
    // Convert to CSC
    A_csc = cholmod_l_triplet_to_sparse(A_coo, A_nnz, cc);

    // Create b as a dense matrix
    b = cholmod_l_allocate_dense(A_m, 1, A_m, CHOLMOD_REAL, cc);
    bx = b->x;
    for (k=0; k<A_m; k++) {
        bx[k] = b_data[k];
    }
    // Make sure the matrix is valid
    if (cholmod_l_check_dense(b, cc) != 1) {
        fprintf(stderr, "ERROR: b vector is not valid");
        return;
    }

    // Calculate Q and R , Q has m by rank and R has rank by m size
    /*E = SuiteSparseQR_C_QR(rank, A_csc, Q, R , cc);*/

    // Return values of x
    xx = x->x;
    for (k=0; k<A_n; k++) {
        x_data[k] = xx[k];
    }

    /* free everything and finish CHOLMOD */
    cholmod_l_free_triplet(&A_coo, cc);
    cholmod_l_free_sparse(&A_csc, cc);
    cholmod_l_free_dense(&x, cc);
    cholmod_l_free_dense(&b, cc);
    cholmod_l_finish(cc);
    return;
}

