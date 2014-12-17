    /*
 *   intcrd.c
 *
 *   thctk - python package for Theoretical Chemistry
 *   Copyright (C) 2004 Christoph Scheurer
 *
 *   This file was taken from the thctk package of Christoph Scheurer.
 *
 *   thctk is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   thctk is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/* {{{ defines and includes */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef THCTK_INTERFACE
#include <Python.h>
#include "thctk.h"
#define vtype   double
#else
#include <stdio.h>
#define vtype   double
#endif

typedef vtype vec[3];
struct ijarray { int *i, *j; };

/* #define ANGEPS          1.0E-7 */
#define ANGEPS          1.0E-12 
#define ONE             1.0
#ifndef MAX
#define MAX(a,b)        (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b)        (((a) < (b)) ? (a) : (b))
#endif
#define DOT(a,b)        ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define NORM(a)         (sqrt(DOT(a,a)))
#define NORMAL(a)       VSCALE(a, ONE/NORM(a));
#define NORMAL2(a,n)    n=NORM(a); VSCALE(a, ONE/n);
#define VCP(a,b)        (a)[0]=(b)[0]; (a)[1]=(b)[1]; (a)[2]=(b)[2];
#define VINV(a,b)       (a)[0]=-(b)[0]; (a)[1]=-(b)[1]; (a)[2]=-(b)[2];
#define VSCALE(a,f)     {register vtype t=(f); (a)[0]*=t; (a)[1]*=t; (a)[2]*=t;}
#define VADDI(a,b)      (a)[0]+=(b)[0]; (a)[1]+=(b)[1]; (a)[2]+=(b)[2];
#define VSUBI(a,b)      (a)[0]-=(b)[0]; (a)[1]-=(b)[1]; (a)[2]-=(b)[2];
#define VADD(c,a,b)     (c)[0]=(a)[0]+(b)[0]; \
                        (c)[1]=(a)[1]+(b)[1]; \
                        (c)[2]=(a)[2]+(b)[2];
#define VSUB(c,a,b)     (c)[0]=(a)[0]-(b)[0]; \
                        (c)[1]=(a)[1]-(b)[1]; \
                        (c)[2]=(a)[2]-(b)[2];
#define CROSS(c,a,b)    (c)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; \
                        (c)[1]=(a)[2]*(b)[0]-(a)[0]*(b)[2]; \
                        (c)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0];
#define DIRV(a,b,e,r)   VSUB(e,b,a); NORMAL2(e,r);
#define AXPY(a,x,y)     (y)[0]+=(a)*(x)[0]; \
                        (y)[1]+=(a)*(x)[1]; \
                        (y)[2]+=(a)*(x)[2];

#define NINTERNALS      7

static int atoms_per_internal[NINTERNALS+1] = {0, 2, 3, 4, 4, 1, 1, 1};

#ifndef THCTK_INTERFACE
void printv(const vec a) { printf("%f %f %f\n", a[0], a[1], a[2]); }
#endif

/* }}} */

/* {{{ internal coordinate values */

THCTK_PRIVATE
double stretch_val(const vec a1, const vec a2) {

    vec r;

    VSUB(r, a1, a2);

    return NORM(r);
}

THCTK_PRIVATE
double bend_val(const vec a1, const vec a2, const vec a3) {

    vec e31, e32;
    vtype c, r31, r32;

    DIRV(a3, a1, e31, r31); /* bond vectors and angles */
    DIRV(a3, a2, e32, r32);
    c = DOT(e31, e32);

    if (c + ANGEPS >  ONE) return 0;
    if (c - ANGEPS < -ONE) return M_PI;
    return acos(c);
}

THCTK_PRIVATE
double out_of_plane_val(const vec a1, const vec a2, const vec a3, const vec a4) {

    vec e41, e42, e43, x1;
    vtype r41, r42, r43, s, c, st;

    DIRV(a4, a1, e41, r41); /* bond vectors and cross products */
    DIRV(a4, a2, e42, r42);
    DIRV(a4, a3, e43, r43);
    CROSS(x1, e42, e43);

    c = DOT(e42, e43);      /* angles and factors */
    s = sqrt(ONE - MIN(ONE, c*c));
    st = DOT(x1, e41)/s;

    if (st + ANGEPS >  ONE) return (M_PI/2);
    if (st - ANGEPS < -ONE) return -(M_PI/2);
    return ((M_PI/2) - acos(st));
}

THCTK_PRIVATE
double torsion_val(const vec a1, const vec a2, const vec a3, const vec a4) {

    vec e12, e23, e34, x1, x2;
    vtype r12, r23, r34, s2, c2, s3, c3, ct;

    DIRV(a1, a2, e12, r12); /* bond vectors and cross products */
    DIRV(a2, a3, e23, r23);
    DIRV(a3, a4, e34, r34);
    CROSS(x1, e12, e23);
    CROSS(x2, e23, e34);

    c2 = - DOT(e12, e23);  /* angles and factors */
    c3 = - DOT(e23, e34);
    s2 = sqrt(ONE - MIN(ONE, c2*c2));
    s3 = sqrt(ONE - MIN(ONE, c3*c3));
    ct = DOT(x1, x2)/(s2*s3);

    if (ct + ANGEPS >  ONE) return 0;
    if (ct - ANGEPS < -ONE) return M_PI;
    if (DOT(x2, e12) < -ANGEPS) return -acos(ct);
    return acos(ct);
}

THCTK_PRIVATE
double cart_val(const vec a1, const int i) {

    return a1[i];
}

THCTK_PRIVATE
double cart_val2(const vec a1,const vec h) {

    return a1[0]*h[0]+a1[1]*h[1]+a1[2]*h[2];
}

#if 0
THCTK_PRIVATE
double dihedral5_val(const vec a1, const vec a2, const vec a3, const vec a4,
    const vec a5) {
    return 0;
}

THCTK_PRIVATE
double linear_bend_val(const vec a1, const vec a2, const vec a3, const vec a4) {
    return 0;
}
#endif

THCTK_PRIVATE
int internals(const vtype *x, const int *cint, double *c) {

    const vtype *p1, *p2, *p3, *p4;
    /* We have to assign to these pointers to ensure the correct ordering of
     * coordinates in the function arguments.
     * ANSI C does not specify the order of evaluation for function arguments!
     */

#define PXYZ    (x+3*((*cint++)-1))
    while (1) {
        register int type = *cint++;
        if (type <= 0) return 0;
        switch (type) {
            case 1:
                p1 = PXYZ; p2 = PXYZ;
                *c++ = stretch_val(p1, p2);
                break;
            case 2:
                p1 = PXYZ; p2 = PXYZ; p3 = PXYZ;
                *c++ = bend_val(p1, p2, p3);
                break;
            case 3:
                p1 = PXYZ; p2 = PXYZ; p3 = PXYZ; p4 = PXYZ;
                *c++ = torsion_val(p1, p2, p3, p4);
                break;
            case 4:
                p1 = PXYZ; p2 = PXYZ; p3 = PXYZ; p4 = PXYZ;
                *c++ = out_of_plane_val(p1, p2, p3, p4);
                break;
            case 5:
                p1 = PXYZ;
                *c++ = cart_val(p1, 0);
                break;
            case 6:
                p1 = PXYZ;
                *c++ = cart_val(p1, 1);
                break;
            case 7:
                p1 = PXYZ;
                *c++ = cart_val(p1, 2);
                break;
            default:
                return 2;  /* unknown internal coordinate type */
        }
    }
    return 1;  /* this point should not be reached! */

#undef PXYZ
}


/* }}} */

THCTK_PRIVATE
int internals_pbc(const vtype *x, const vtype *h, const int *cint, double *c) {

    const vtype *p1, *p2, *p3, *p4;
    /* We have to assign to these pointers to ensure the correct ordering of
     * coordinates in the function arguments.
     * ANSI C does not specify the order of evaluation for function arguments!
     */

#define PXYZ    (x+3*((*cint++)-1))
    while (1) {
        register int type = *cint++;
        if (type <= 0) return 0;
        switch (type) {
            case 1:
                p1 = PXYZ; p2 = PXYZ;
                *c++ = stretch_val(p1, p2);
                break;
            case 2:
                p1 = PXYZ; p2 = PXYZ; p3 = PXYZ;
                *c++ = bend_val(p1, p2, p3);
                break;
            case 3:
                p1 = PXYZ; p2 = PXYZ; p3 = PXYZ; p4 = PXYZ;
                *c++ = torsion_val(p1, p2, p3, p4);
                break;
            case 4:
                p1 = PXYZ; p2 = PXYZ; p3 = PXYZ; p4 = PXYZ;
                *c++ = out_of_plane_val(p1, p2, p3, p4);
                break;
            case 5:
                p1 = PXYZ;
                *c++ = cart_val2(p1, h);
                break;
            case 6:
                p1 = PXYZ;
                *c++ = cart_val2(p1, h+3);
                break;
            case 7:
                p1 = PXYZ;
                *c++ = cart_val2(p1, h+6);
                break;
            default:
                return 2;  /* unknown internal coordinate type */
        }
    }
    return 1;  /* this point should not be reached! */

#undef PXYZ
}
/* }}} */


/* {{{ B-matrix elements */

THCTK_PRIVATE
int stretch(const vec a1, const vec a2, vec s1, vec s2) {

    vtype r;

    DIRV(a1, a2, s2, r)
    VINV(s1, s2);
                
    return 0;
}

THCTK_PRIVATE
int bend(const vec a1, const vec a2, const vec a3,
    vec s1, vec s2, vec s3) {

    vec e31, e32;
    vtype c, s, r31, r32;

    DIRV(a3, a1, e31, r31); /* bond vectors and angles */
    DIRV(a3, a2, e32, r32);
    c = DOT(e31, e32);
    s = sqrt(ONE - MIN(ONE, c*c));

    VINV(s1, e32);          /* s1 = (c*e31 - e32)/(r31*s) */
    AXPY(c, e31, s1);
    VSCALE(s1, ONE/(r31*s));

    VINV(s2, e31);          /* s2 = (c*e32 - e31)/(r32*s) */
    AXPY(c, e32, s2);
    VSCALE(s2, ONE/(r32*s));

    VINV(s3, s1);           /* s3 = - s1 - s2 */
    VSUBI(s3, s2);

    return 0;
}

THCTK_PRIVATE
int out_of_plane(const vec a1, const vec a2, const vec a3, const vec a4,
    vec s1, vec s2, vec s3, vec s4) {

    vec e41, e42, e43, x1, x2, x3;
    vtype r41, r42, r43, s, c, st, ct, tt, a, b;

    DIRV(a4, a1, e41, r41); /* bond vectors and cross products */
    DIRV(a4, a2, e42, r42);
    DIRV(a4, a3, e43, r43);
    CROSS(x1, e42, e43);
    CROSS(x2, e43, e41);
    CROSS(x3, e41, e42);

    c = DOT(e42, e43);      /* angles and factors */
    s = sqrt(ONE - MIN(ONE, c*c));
    st = DOT(x1, e41)/s;
    ct = sqrt(ONE - MIN(ONE, st*st));
    tt = st/ct;
    a = ONE/(ct*s);
    b = -tt/(s*s);

    VINV(s1, e41);          /* s1 */
    VSCALE(s1, tt);
    AXPY(a, x1, s1);
    VSCALE(s1, ONE/r41);

    VCP(s2, e42);           /* s2 */
    AXPY(-c, e43, s2);
    VSCALE(s2, b);
    AXPY(a, x2, s2);
    VSCALE(s2, ONE/r42);

    VCP(s3, e43);           /* s3 */
    AXPY(-c, e42, s3);
    VSCALE(s3, b);
    AXPY(a, x3, s3);
    VSCALE(s3, ONE/r43);

    VINV(s4, s1);           /* s4 = - s1 - s2 - s3 */
    VSUBI(s4, s2);
    VSUBI(s4, s3);

    return 0;
}

THCTK_PRIVATE
int torsion(const vec a1, const vec a2, const vec a3, const vec a4,
    vec s1, vec s2, vec s3, vec s4) {

    vec e12, e23, e34, x1, x2;
    vtype r12, r23, r34, sp2, cp2, sp3, cp3, f;

    DIRV(a1, a2, e12, r12); /* bond vectors and cross products */
    DIRV(a2, a3, e23, r23);
    DIRV(a3, a4, e34, r34);
    CROSS(x1, e12, e23);
    CROSS(x2, e23, e34);

    cp2 = - DOT(e12, e23);  /* angles and factors */
    cp3 = - DOT(e23, e34);
    sp2 = sqrt(ONE - MIN(ONE, cp2*cp2));
    sp3 = sqrt(ONE - MIN(ONE, cp3*cp3));
    f = r12*sp2*sp2;

    VINV(s1, x1);           /* s1 */
    VSCALE(s1, ONE/f);

    VCP(s4, x2);            /* s4 */
    VSCALE(s4, ONE/(r34*sp3*sp3));

    VCP(s2, x1);           /* s2 */
    VSCALE(s2, (r23 - r12*cp2)/(r23*f));
    AXPY(-cp3/(r23*sp3*sp3), x2, s2);

    VINV(s3, s1);           /* s3 = - s1 - s2 - s4 */
    VSUBI(s3, s2);
    VSUBI(s3, s4);

    return 0;
}

#if 0
THCTK_PRIVATE
int dihedral5(const vec a1, const vec a2, const vec a3, const vec a4,
    const vec a5, vec s1, vec s2, vec s3, vec s4, vec s5) {
    return 0;
}

THCTK_PRIVATE
int linear_bend(const vec a1, const vec a2, const vec a3, const vec a4,
    vec s1, vec s2, vec s3, vec s4) {
    return 0;
}
#endif

/* }}} */

/* {{{ sparse B-matrix functions */

/* {{{ sorting functions */

THCTK_PRIVATE
void bsort(int *jb, int n) {

    int i, j, jt;

    for (i=0; i<n-1; i++) {
        for (j=0; j<n-1-i; j++)
            if (jb[j+1] < jb[j]) {  /* compare the two neighbors */
            jt = jb[j]; jb[j] = jb[j+1]; jb[j+1] = jt;  /* swap if necessary */
        }
    }
}

THCTK_PRIVATE
void bsort2(vtype *b, int *jb, int n) {

    int i, j, jt;
    vtype t;

    for (i=0; i<n-1; i++) {
        for (j=0; j<n-1-i; j++)
            if (jb[j+1] < jb[j]) {  /* compare the two neighbors */
            jt = jb[j]; jb[j] = jb[j+1]; jb[j+1] = jt;  /* swap column indices */
             t = b[j];   b[j] = b[j+1];   b[j+1] = t;   /* swap matrix elements */
        }
    }
}

/* }}} */

/* {{{ symbolicAc */

#define ERREXIT(i)  { *errcode=i; goto exit; }

THCTK_PRIVATE
struct ijarray *symbolicAc(const int *cint, int natom, int *errcode) {

    /*
     * compute the symbolic sparse structure of the matrix Ac = Bt*B i.e. the
     * connectivity between cartesian coordinates induced by the internals
     */

    int *at=NULL, *jat=NULL, *jc=NULL, *row=NULL, *c, type, i, n, nint;
    static struct ijarray conn;
    struct ijarray *retval=NULL;

    *errcode = 0;
    /* atoms in cint are indexed with base 1, thus (natom+1) elements */
    if (! (jat = (int *) calloc((natom+1), sizeof(int)))) ERREXIT(1);
    
    /* count in how many internals each atom is involved */
    nint = 0;
    c = (int *) cint;
    while (1) {
        type = *c++;
        if (type > NINTERNALS) ERREXIT(2);  /* unknown internal coordinate type */
        if (type <= 0) break;               /* end of list, everything went fine */
        nint++;
        for (i=0; i<atoms_per_internal[type]; i++) jat[*c++]++;
    }

    /* allocate at and generate pointers into it */
    jat[0] = 0;
    for (i=0; i<natom; i++) jat[i+1] += jat[i];

    if (! (at = (int *) malloc(jat[natom]*sizeof(int)))) ERREXIT(3);

    /* jc contains pointers to the beginning of each internal coordinate in cint */
    if (! (jc = (int *) malloc((nint+1)*sizeof(int)))) ERREXIT(4);

    if (! (conn.i = (int *) calloc((natom+1), sizeof(int)))) ERREXIT(5);
    
    /* reverse the assignment of cint and create pointers into cint */
    nint = 0;
    c = (int *) cint;
    n = 0;
    while (1) {
        int na;
        type = *c++;
        if (type <= 0) break;       /* end of list, everything went fine */
        jc[nint] = n++;
        na = atoms_per_internal[type];
        for (i=0; i<na; i++, c++, n++) {
            register int p=(*c)-1;  /* atom index in cint starts with 1 */
            at[jat[p]++] = nint;
            conn.i[p] += na;        /* how many other atoms are maximally connected to p */
        }
        nint++;
    }
    jc[nint] = n;

    for (i=natom; i>0; i--) jat[i] = jat[i-1];  /* shift offsets back */
    jat[0] = 0;


    n = -1;
    for (i=0; i<natom; i++) n = MAX(n, conn.i[i]);

    if (! (row = (int *) malloc(n*sizeof(int)))) ERREXIT(6);

    for (i=0; i<natom; i++) {
        int k, l;
        n = 0;
        for (k=jat[i]; k<jat[i+1]; k++) {
            for (l=jc[at[k]]+1; l<jc[at[k]+1]; l++)
                row[n++] = cint[l]-1;
        }
        bsort(row, n);
        conn.i[i+1] = 0;
        k = -1;
        for (l=0; l<n; l++) {
            register int j = row[l];
            if (j > k && j != i) {
                k = j;
                conn.i[i+1]++;
            }
        }
    }

    conn.i[0] = 0;
    for (i=0; i<natom; i++) conn.i[i+1] += conn.i[i];
    n = conn.i[natom];

    if (! (conn.j = (int *) malloc(n*sizeof(int)))) ERREXIT(7);

    /*
     * find all other atoms each atom is connected to by an internal
     * coordinate, i.e. at -> cint -> at
     */

    for (i=0; i<natom; i++) {
        int k, l;
        n = 0;
        for (k=jat[i]; k<jat[i+1]; k++) {
            for (l=jc[at[k]]+1; l<jc[at[k]+1]; l++)
                row[n++] = cint[l]-1;
        }
        bsort(row, n);
        k = -1;
        for (l=0; l<n; l++) {
            register int j = row[l];
            if (j > k && j != i) {
                k = j;
                conn.j[conn.i[i]++] = j;
            }
        }
    }

    for (i=natom; i>0; i--) conn.i[i] = conn.i[i-1];  /* shift offsets back */
    conn.i[0] = 0;

    retval = &conn;

exit:
    if (jat) free(jat);
    if (at) free(at);
    if (jc) free(jc);
    if (row) free(row);
    if (!retval) {
        if (conn.j) free(conn.j);
        if (conn.i) free(conn.i);
    }

    return retval;
}

/* }}} */

/* {{{ conn2crd_p */

THCTK_PRIVATE
struct ijarray *conn2crd_p(int natom, int diag, const int *cj, const int *ci,
    const int *p, int sort, int *errcode) {

    int i, j, ai, aj, nx=3*natom, nnz=0;
    int *pp=NULL;

    static struct ijarray bb;
    struct ijarray *retval=NULL;

    *errcode = 0;

    if (! (pp = (int *) malloc(nx*sizeof(int)))) ERREXIT(1);

    for (i=0; i<nx; i++) pp[p[i]] = i;  /* inverse of permutation p */

    if (! (bb.i = (int *) calloc((nx+1), sizeof(int)))) ERREXIT(2);

    for (ai=0; ai<natom; ai++) {
        int m, k, l;
        aj = ai;    /* diagonal block */
        for (k=0; k<3; k++) {
            i = p[3*ai+k];
            for (l=0; l<3; l++) {
                j = p[3*aj+l];
                if (j > i) bb.i[i]++;
                else if (j==i && !diag) bb.i[i]++;
            }
        }
        for (m=ci[ai]; m<ci[ai+1]; m++) {   /* connected atoms */
            aj = cj[m];
            for (k=0; k<3; k++) {
                i = p[3*ai+k];
                for (l=0; l<3; l++) {
                    j = p[3*aj+l];
                    if (j > i) bb.i[i]++;
                    else if (j==i && !diag) bb.i[i]++;
                }
            }
        }
    }

    for (i=0; i<nx; i++) {
        int n = bb.i[i];
        bb.i[i] = nnz;
        nnz += n;
    }
    bb.i[nx] = nnz;

    if (! (bb.j = (int *) malloc(nnz*sizeof(int)))) ERREXIT(3);

    for (ai=0; ai<natom; ai++) {
        int m, k, l;
        aj = ai;    /* diagonal block */
        for (k=0; k<3; k++) {
            i = p[3*ai+k];
            for (l=0; l<3; l++) {
                j = p[3*aj+l];
                if (j > i) bb.j[bb.i[i]++] = j;
                else if (j==i && !diag) bb.j[bb.i[i]++] = j;
            }
        }
        for (m=ci[ai]; m<ci[ai+1]; m++) {   /* connected atoms */
            aj = cj[m];
            for (k=0; k<3; k++) {
                i = p[3*ai+k];
                for (l=0; l<3; l++) {
                    j = p[3*aj+l];
                    if (j > i) bb.j[bb.i[i]++] = j;
                    else if (j==i && !diag) bb.j[bb.i[i]++] = j;
                }
            }
        }
    }
    
    for (i=nx; i>0; i--) bb.i[i] = bb.i[i-1];  /* shift offsets back */
    bb.i[0] = 0;

    if (sort) {
        for (i=0; i<nx; i++) bsort(&bb.j[bb.i[i]], bb.i[i+1]-bb.i[i]);
    }

    retval = &bb;

exit:
    if (pp) free(pp);
    if (!retval) {
        if (bb.j) free(bb.j);
        if (bb.i) free(bb.i);
    }

    return retval;
}

#undef ERREXIT

/* }}} */

/* {{{ conn2crd */

THCTK_PRIVATE
int conn2crd(int natom, int diag, const int *cj, const int *ci,
             int *aj, int *ai) {

    int i, j, nnz=0;
    int *a0, *a1, *a2=aj;

    for (i=0; i<natom; i++) {
        int m=3*i, n=0;
        for (j=ci[i]; j<ci[i+1]; j++) if (cj[j]>i) n += 3;
        if (diag) { /* diagonal is stored separately */
            *ai++ = nnz; nnz += n + 2; a0 = a2;         /* x row */
            *ai++ = nnz; nnz += n + 1; a1 = a0 + n + 2; /* y row */
            *ai++ = nnz; nnz += n    ; a2 = a1 + n + 1; /* z row */
            *a0++ = m + 1;  *a0++ = m + 2;              /* diagonal block */
                            *a1++ = m + 2;
        } else {
            *ai++ = nnz; nnz += n + 3; a0 = a2;         /* x row */
            *ai++ = nnz; nnz += n + 2; a1 = a0 + n + 3; /* y row */
            *ai++ = nnz; nnz += n + 1; a2 = a1 + n + 2; /* z row */
            *a0++ = m;  *a0++ = m + 1;  *a0++ = m + 2;  /* diagonal block */
                        *a1++ = m + 1;  *a1++ = m + 2;
                                        *a2++ = m + 2;
        }
        for (j=ci[i]; j<ci[i+1]; j++) {
            if (cj[j]>i) {
                register int k=3*cj[j];
                *a0++ = k; *a1++ = k; *a2++ = k++; /* x column */
                *a0++ = k; *a1++ = k; *a2++ = k++; /* y column */
                *a0++ = k; *a1++ = k; *a2++ = k++; /* z column */
            }
        }
    }
    *ai = nnz;

    return 0;
}

/* }}} */

/* {{{ qr_solve */
/*THCTK_PRIVATE*/
/*int qr_solve(double const int *cint, int *nint, int *nnz) {*/

    /*int type, n;*/

    /**nint = 0;*/
    /**nnz = 0;*/

    /*while (1) {*/
        /*type = *cint++;*/
        /*if (type > NINTERNALS) return 1;    [> unknown internal coordinate type <]*/
        /*if (type <= 0) return 0;            [> end of list, everything went fine <]*/
        /*n = atoms_per_internal[type];*/
        /*cint += n;*/
        /*(*nnz) += 3*n;*/
        /*(*nint)++;*/
    /*}*/
/*}*/

/* }}} */

/* {{{ Bmatrix_nnz */

THCTK_PRIVATE
int Bmatrix_nnz(const int *cint, int *nint, int *nnz) {

    int type, n;

    *nint = 0;
    *nnz = 0;

    while (1) {
        type = *cint++;
        if (type > NINTERNALS) return 1;    /* unknown internal coordinate type */
        if (type <= 0) return 0;            /* end of list, everything went fine */
        n = atoms_per_internal[type];
        (*nnz) += 3*n;
        cint += n;
        (*nint)++;
    }
}

/* }}} */

/* {{{ Bmatrix_pbc_nnz */

THCTK_PRIVATE
int Bmatrix_pbc_nnz(int nx, const int *cint, int *nint, int *nnz) {

    int type, n, na;
    int i1, i2, i3, i4, z,nnzsum;
    bool txset, tyset, tzset;

    *nint = 0;
    *nnz = 0;

    na = nx/3;

    while (1) {
        type = *cint++;
        if (type > NINTERNALS) return 1;    /* unknown internal coordinate type */
        if (type <= 0) {
            return 0;            /* end of list, everything went fine */
        }
        n = atoms_per_internal[type];
        //every type gets 3*n elements
        nnzsum = 0; nnzsum += 3*n;
        txset=false; tyset=false; tzset=false;
        switch (type) {
            case 1:
                i1 = (*cint++)-1; i2 = (*cint++)-1;
                if (i1%na==i2%na) {nnzsum-=3;}
                z = i1/na; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i2/na; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                if (txset) {nnzsum += 3;}
                if (tyset) {nnzsum += 3;}
                if (tzset) {nnzsum += 3;}
                break;
            case 2:
                i1 = (*cint++)-1; i2 = (*cint++)-1; i3 = (*cint++)-1;
                if (i1%na==i2%na && i2%na==i3%na) {nnzsum -= 6;}
                else if (i1%na==i2%na || i2%na==i3%na || i1%na==i3%na) {nnzsum -= 3;}
                else {  } //nothing
                z = i1/na; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i2/na; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i3/na; /*p3*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                if (txset) {nnzsum += 3;}
                if (tyset) {nnzsum += 3;}
                if (tzset) {nnzsum += 3;}
                break;
            case 3:
                i1 = (*cint++)-1; i2 = (*cint++)-1; i3 = (*cint++)-1; i4 = (*cint++)-1;
                if (i1%na==i2%na && i2%na==i3%na && i3%na==i4%na) {nnzsum -= 9;}
                else if ( (i1%na==i2%na && i2%na==i3%na) || \
                          (i1%na==i2%na && i2%na==i4%na) || \
                          (i2%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i2%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i2%na==i4%na) || \
                          (i1%na==i4%na && i2%na==i3%na) \
                          ) {nnzsum -= 6;}
                else if (i1%na==i2%na || i1%na==i3%na || i1%na==i4%na || \
                    i2%na==i3%na || i2%na==i4%na || i3%na==i4%na ) {nnzsum -= 3;}
                else { } //nothing
                z = i1/na; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i2/na; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i3/na; /*p3*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i4/na; /*p4*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                if (txset) {nnzsum += 3;}
                if (tyset) {nnzsum += 3;}
                if (tzset) {nnzsum += 3;}
                break;
            case 4:
                i1 = (*cint++)-1; i2 = (*cint++)-1; i3 = (*cint++)-1; i4 = (*cint++)-1;
                if (i1%na==i2%na && i2%na==i3%na && i3%na==i4%na) {nnzsum -= 9;}
                else if ( (i1%na==i2%na && i2%na==i3%na) || \
                          (i1%na==i2%na && i2%na==i4%na) || \
                          (i2%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i2%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i2%na==i4%na) || \
                          (i1%na==i4%na && i2%na==i3%na) \
                          ) {nnzsum -= 6;}
                else if (i1%na==i2%na || i1%na==i3%na || i1%na==i4%na || \
                    i2%na==i3%na || i2%na==i4%na || i3%na==i4%na ) {nnzsum -= 3;}
                else {} //nothing
                z = i1/na; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i2/na; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i3/na; /*p3*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                z = i4/na; /*p4*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;}
                if (txset) {nnzsum += 3;}
                if (tyset) {nnzsum += 3;}
                if (tzset) {nnzsum += 3;}
                break;
            case 5:
                i1 = (*cint++)-1;
                break;
            case 6:
                i1 = (*cint++)-1;
                break;
            case 7:
                i1 = (*cint++)-1;
                break;
            default:
                break;
        }
        /*cint += n; */
        (*nnz) += nnzsum;
        (*nint)++;
    }
}

/* }}} */

/* {{{ Bmatrix_pbc2_nnz */

THCTK_PRIVATE
int Bmatrix_pbc2_nnz(int nx, const int *cint, int *nint, int *nnz) {

    int type, n, na;
    int i1, i2, i3, i4, nnzsum;

    *nint = 0;
    *nnz = 0;

    na = nx/3;

    while (1) {
        type = *cint++;
        if (type > NINTERNALS) return 1;    /* unknown internal coordinate type */
        if (type <= 0) {
            return 0;            /* end of list, everything went fine */
        }
        n = atoms_per_internal[type];
        //every type gets 3*n elements
        nnzsum = 0; nnzsum += 3*n;
        switch (type) {
            case 1:
                i1 = (*cint++)-1; i2 = (*cint++)-1;
                if (i1%na==i2%na) {nnzsum-=3;}
                nnzsum += 9;
                break;
            case 2:
                i1 = (*cint++)-1; i2 = (*cint++)-1; i3 = (*cint++)-1;
                if (i1%na==i2%na && i2%na==i3%na) {nnzsum -= 6;}
                else if (i1%na==i2%na || i2%na==i3%na || i1%na==i3%na) {nnzsum -= 3;}
                else {  } //nothing
                nnzsum += 9;
                break;
            case 3:
                i1 = (*cint++)-1; i2 = (*cint++)-1; i3 = (*cint++)-1; i4 = (*cint++)-1;
                if (i1%na==i2%na && i2%na==i3%na && i3%na==i4%na) {nnzsum -= 9;}
                else if ( (i1%na==i2%na && i2%na==i3%na) || \
                          (i1%na==i2%na && i2%na==i4%na) || \
                          (i2%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i2%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i2%na==i4%na) || \
                          (i1%na==i4%na && i2%na==i3%na) \
                          ) {nnzsum -= 6;}
                else if (i1%na==i2%na || i1%na==i3%na || i1%na==i4%na || \
                    i2%na==i3%na || i2%na==i4%na || i3%na==i4%na ) {nnzsum -= 3;}
                else { } //nothing
                nnzsum += 9;
                break;
            case 4:
                i1 = (*cint++)-1; i2 = (*cint++)-1; i3 = (*cint++)-1; i4 = (*cint++)-1;
                if (i1%na==i2%na && i2%na==i3%na && i3%na==i4%na) {nnzsum -= 9;}
                else if ( (i1%na==i2%na && i2%na==i3%na) || \
                          (i1%na==i2%na && i2%na==i4%na) || \
                          (i2%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i3%na==i4%na) || \
                          (i1%na==i2%na && i3%na==i4%na) || \
                          (i1%na==i3%na && i2%na==i4%na) || \
                          (i1%na==i4%na && i2%na==i3%na) \
                          ) {nnzsum -= 6;}
                else if (i1%na==i2%na || i1%na==i3%na || i1%na==i4%na || \
                    i2%na==i3%na || i2%na==i4%na || i3%na==i4%na ) {nnzsum -= 3;}
                else {} //nothing
                nnzsum += 9;
                break;
            case 5:
                i1 = (*cint++)-1;
                nnzsum += 9;
                break;
            case 6:
                i1 = (*cint++)-1;
                nnzsum += 9;
                break;
            case 7:
                i1 = (*cint++)-1;
                nnzsum += 9;
                break;
            default:
                break;
        }
        /*cint += n; */
        (*nnz) += nnzsum;
        (*nint)++;
    }
}

/* }}} */

/* {{{ Bmatrix*/

THCTK_PRIVATE
int Bmatrix(const vtype *x, const int *cint, vtype *b, int *jb, int *ib,
    int sort) {

    int type, nint = 0, n, *j;
    int p1, p2, p3, p4, nb = 0;

#define NXYZ    (3*((*cint++)-1))
#define BIDX(p) *jb++ = p; *jb++ = ++p; *jb++ = ++p;

    while (1) {
        type = *cint++;
        ib[nint] = nb;      /* nint iterates over the rows of B */
        j = jb;
        if (type <= 0) return 0;
        switch (type) {
            case 1:
                p1 = NXYZ; p2 = NXYZ;
                stretch(x+p1, x+p2, b, b+3);
                BIDX(p1); BIDX(p2);
                break;
            case 2:
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ;
                bend(x+p1, x+p2, x+p3, b, b+3, b+6);
                BIDX(p1); BIDX(p2); BIDX(p3);
                break;
            case 3:
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ; p4 = NXYZ;
                torsion(x+p1, x+p2, x+p3, x+p4, b, b+3, b+6, b+9);
                BIDX(p1); BIDX(p2); BIDX(p3); BIDX(p4);
                break;
            case 4:
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ; p4 = NXYZ;
                out_of_plane(x+p1, x+p2, x+p3, x+p4, b, b+3, b+6, b+9);
                BIDX(p1); BIDX(p2); BIDX(p3); BIDX(p4);
                break;
            case 5:
                /* adding single atoms for cart. constraints */
                p1 = NXYZ; 
                *(b+0) = 1.0; *(b+1) = 0.0; *(b+2) = 0.0; 
                BIDX(p1); 
                break;
            case 6:
                /* adding single atoms for cart. constraints */
                p1 = NXYZ; 
                *(b+0) = 0.0; *(b+1) = 1.0; *(b+2) = 0.0; 
                BIDX(p1); 
                break;
            case 7:
                /* adding single atoms for cart. constraints */
                p1 = NXYZ; 
                *(b+0) = 0.0; *(b+1) = 0.0; *(b+2) = 1.0; 
                BIDX(p1); 
                break;
            default:
                return 2;  /* unknown internal coordinate type */
        }
        n = 3*atoms_per_internal[type];
        if (sort) bsort2(b, j, n); /* ensure that the column index is increasing */
        b += n;
        nb += n;
        nint++;
    }
    return 1;  /* this point should not be reached! */

#undef NXYZ
#undef BIDX
}

/* }}} */

/* {{{ Bmatrix_pbc*/

THCTK_PRIVATE
int Bmatrix_pbc(int nx, const vtype *x, const int *cint, vtype *b, int *jb, int *ib, int sort) {

    int type, nint = 0, n, *j;
    int p1, p2, p3, p4, nb = 0, pad = 0;
    int i1, i2, i3, i4;
    int z;
    bool txset, tyset, tzset;
    vec dsum, d1, d2, d3, d4, zero, tx, ty, tz;
    zero[0]=0.0; zero[1]=0.0, zero[2]=0.0;

#define NXYZ    (3*((*cint++)-1))
#define BIDX(p) *jb++ = p%nx; *jb++ = (++p)%nx; *jb++ = (++p)%nx;

#define SETVEC(vec1, num) vec1[0]= (*(b+num+0)); \
                          vec1[1]= (*(b+num+1)); \
                          vec1[2]= (*(b+num+2));
#define SETB(num, vec1)   (*(b+num+0)) = vec1[0]; \
                          (*(b+num+1)) = vec1[1]; \
                          (*(b+num+2)) = vec1[2]; \

#define BIDT(num) *jb++ = nx+num; *jb++ = nx+num+1; *jb++ = nx+num+2;

    /* for atoms that lie outside the cell we need to add components 
     * to the back-folded indices and also to the lattice vectors  */

    while (1) {
        type = *cint++;
        ib[nint] = nb;      /* nint iterates over the rows of B */
        j = jb;
        if (type <= 0) {
            return 0;
        }
        switch (type) {
            case 1:
                pad = 0;
                p1 = NXYZ; p2 = NXYZ;
                i1=p1; i2=p2;
                stretch(x+p1, x+p2, d1, d2);
                // atoms are the same in different cells 
                // so we only set one
                if (i1%nx==i2%nx) {
                    SETB(0,zero); 
                    // pass on index of smallest number
                    BIDX(p1);
                    pad -= 3;
                }
                else {
                    SETB(0,d1);SETB(3,d2); 
                    BIDX(p1); BIDX(p2);
                }
                // set unit cell values
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                txset = false; tyset = false; tzset = false;
                z = i1/nx; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d1);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d1);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d1);}
                z = i2/nx; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d2);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d2);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d2);}
                if (txset) {pad += 3; SETB(3+pad,tx); BIDT(0);}
                if (tyset) {pad += 3; SETB(3+pad,ty); BIDT(3);}
                if (tzset) {pad += 3; SETB(3+pad,tz); BIDT(6);}
                break;
            case 2: // bending angles
                pad = 0;
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ;
                i1=p1; i2=p2; i3=p3;
                bend(x+p1, x+p2, x+p3, d1, d2, d3);
                if (i1%nx==i2%nx && i2%nx==i3%nx) {
                    SETB(0,zero);
                    //pass on smallest index
                    BIDX(p1);
                    pad -= 6;
                }
                else if (i1%nx==i2%nx || i1%nx==i3%nx || i2%nx==i3%nx ) {
                    if (i1%nx==i2%nx) {
                        VADD(dsum,d1,d2);
                        SETB(0,dsum); SETB(3,d3);
                        BIDX(p1); BIDX(p3);
                        pad -= 3;
                    }
                    else if (i2%nx==i3%nx) {
                        VADD(dsum,d2,d3);
                        SETB(0,d1); SETB(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 3;
                    }
                    else {
                        VADD(dsum,d1,d3);
                        SETB(0,dsum); SETB(3,d2);
                        BIDX(p1); BIDX(p2);
                        pad -= 3;
                    }
                }
                else { //default, no duplicate atoms
                    SETB(0,d1); SETB(3,d2); SETB(6,d3);
                    BIDX(p1); BIDX(p2); BIDX(p3);
                }
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                txset = false; tyset = false; tzset = false;
                z = i1/nx; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d1);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d1);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d1);}
                z = i2/nx; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d2);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d2);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d2);}
                z = i3/nx; /*p3*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d3);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d3);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d3);}
                if (txset) {pad += 3; SETB(6+pad,tx); BIDT(0);}
                if (tyset) {pad += 3; SETB(6+pad,ty); BIDT(3);}
                if (tzset) {pad += 3; SETB(6+pad,tz); BIDT(6);}
                break;
            case 3: // torsions
                pad = 0;
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ; p4 = NXYZ;
                i1=p1; i2=p2;i3=p3;i4=p4;
                torsion(x+p1, x+p2, x+p3, x+p4, d1, d2, d3, d4);
                if (i1%nx==i2%nx && i2%nx==i3%nx && i3%nx==i4%nx) {
                    // all indices are the same
                    SETB(0,zero);
                    //pass on smallest index
                    BIDX(p1);
                    pad -= 9;
                }
                else if ( (i1%nx==i2%nx && i2%nx==i3%nx) || \
                          (i1%nx==i2%nx && i2%nx==i4%nx) || \
                          (i2%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i2%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i2%nx==i4%nx) || \
                          (i1%nx==i4%nx && i2%nx==i3%nx) \
                          ) {
                    // three indices are the same
                    if (i1%nx==i2%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d3);
                        SETB(0,dsum); SETB(3,d4);
                        BIDX(p1); BIDX(p4);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d4);
                        SETB(0,dsum); SETB(3,d3);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i2%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d2,d3); VADD(dsum,dsum,d4);
                        SETB(3,d1); SETB(0,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d3); VADD(dsum,dsum,d4);
                        SETB(0,dsum); SETB(3,d2);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d2);
                        SETB(0,dsum); 
                        VADD(dsum,d3,d4);
                        SETB(3,dsum);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d3);
                        SETB(0,dsum); 
                        VADD(dsum,d2,d4);
                        SETB(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else  { //(i1%nx==i4%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d4);
                        SETB(0,dsum); 
                        VADD(dsum,d2,d3);
                        SETB(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                }
                else if (i1%nx==i2%nx || i1%nx==i3%nx || i1%nx==i4%nx || \
                         i2%nx==i3%nx || i2%nx==i4%nx || i3%nx==i4%nx ) {
                    // two indices are the same
                    if (i1%nx==i2%nx) {
                        VADD(dsum,d1,d2);
                        SETB(0,dsum); SETB(3,d3); SETB(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i3%nx) {
                        VADD(dsum,d1,d3);
                        SETB(0,dsum); SETB(3,d2); SETB(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i4%nx) {
                        VADD(dsum,d1,d4);
                        SETB(0,dsum); SETB(3,d2); SETB(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else if (i2%nx==i3%nx) {
                        VADD(dsum,d2,d3);
                        SETB(0,d1); SETB(3,dsum); SETB(6,d4);
                        BIDX(p1); BIDX(p2); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i2%nx==i4%nx) {
                        VADD(dsum,d2,d4);
                        SETB(0,d1); SETB(3,dsum); SETB(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else {
                        VADD(dsum,d3,d4);
                        SETB(0,d1); SETB(3,d2); SETB(6,dsum);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                }
                else { //default, no duplicate atoms
                    // no indices are the same
                    SETB(0,d1); SETB(3,d2); SETB(6,d3); SETB(9,d4);
                    BIDX(p1); BIDX(p2); BIDX(p3); BIDX(p4);
                }
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                txset = false; tyset = false; tzset = false;
                z = i1/nx; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d1);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d1);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d1);}
                z = i2/nx; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d2);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d2);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d2);}
                z = i3/nx; /*p3*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d3);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d3);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d3);}
                z = i4/nx; /*p4*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d4);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d4);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d4);}
                if (txset) {pad += 3; SETB(9+pad,tx); BIDT(0);}
                if (tyset) {pad += 3; SETB(9+pad,ty); BIDT(3);}
                if (tzset) {pad += 3; SETB(9+pad,tz); BIDT(6);}
                break;
            case 4:
                pad = 0;
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ; p4 = NXYZ;
                i1=p1;i2=p2;i3=p3;i4=p4;
                out_of_plane(x+p1, x+p2, x+p3, x+p4, d1, d2, d3, d4);
                if (i1%nx==i2%nx && i2%nx==i3%nx && i3%nx==i4%nx) {
                    // all indices are the same
                    SETB(0,zero);
                    //pass on smallest index
                    BIDX(p1);
                    pad -= 9;
                }
                else if ( (i1%nx==i2%nx && i2%nx==i3%nx) || \
                          (i1%nx==i2%nx && i2%nx==i4%nx) || \
                          (i2%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i2%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i2%nx==i4%nx) || \
                          (i1%nx==i4%nx && i2%nx==i3%nx) \
                          ) {
                    // three indices are the same
                    if (i1%nx==i2%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d3);
                        SETB(0,dsum); SETB(3,d4);
                        BIDX(p1); BIDX(p4);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d4);
                        SETB(0,dsum); SETB(3,d3);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i2%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d2,d3); VADD(dsum,dsum,d4);
                        SETB(3,d1); SETB(0,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d3); VADD(dsum,dsum,d4);
                        SETB(0,dsum); SETB(3,d2);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d2);
                        SETB(0,dsum); 
                        VADD(dsum,d3,d4);
                        SETB(3,dsum);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d3);
                        SETB(0,dsum); 
                        VADD(dsum,d2,d4);
                        SETB(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else { //(i1%nx==i4%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d4);
                        SETB(0,dsum); 
                        VADD(dsum,d2,d3);
                        SETB(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                }
                else if (i1%nx==i2%nx || i1%nx==i3%nx || i1%nx==i4%nx || \
                         i2%nx==i3%nx || i2%nx==i4%nx || i3%nx==i4%nx ) {
                    // two indices are the same
                    if (i1%nx==i2%nx) {
                        VADD(dsum,d1,d2);
                        SETB(0,dsum); SETB(3,d3); SETB(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i3%nx) {
                        VADD(dsum,d1,d3);
                        SETB(0,dsum); SETB(3,d2); SETB(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i4%nx) {
                        VADD(dsum,d1,d4);
                        SETB(0,dsum); SETB(3,d2); SETB(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else if (i2%nx==i3%nx) {
                        VADD(dsum,d2,d3);
                        SETB(0,d1); SETB(3,dsum); SETB(6,d4);
                        BIDX(p1); BIDX(p2); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i2%nx==i4%nx) {
                        VADD(dsum,d2,d4);
                        SETB(0,d1); SETB(3,dsum); SETB(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else {
                        VADD(dsum,d3,d4);
                        SETB(0,d1); SETB(3,d2); SETB(6,dsum);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                }
                else { //default, no duplicate atoms
                    // no indices are the same
                    SETB(0,d1); SETB(3,d2); SETB(6,d3); SETB(9,d4);
                    BIDX(p1); BIDX(p2); BIDX(p3); BIDX(p4);
                }
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                txset = false; tyset = false; tzset = false;
                z = i1/nx; /*p1*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d1);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d1);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d1);}
                z = i2/nx; /*p2*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d2);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d2);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d2);}
                z = i3/nx; /*p3*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d3);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d3);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d3);}
                z = i4/nx; /*p4*/
                if ( z==1 || z==4 || z==5 || z==7) {txset=true;VADD(tx,tx, d4);}
                if ( z==2 || z==4 || z==6 || z==7) {tyset=true;VADD(ty,ty, d4);}
                if ( z==3 || z==5 || z==6 || z==7) {tzset=true;VADD(tz,tz, d4);}
                if (txset) {pad += 3; SETB(9+pad,tx); BIDT(0);}
                if (tyset) {pad += 3; SETB(9+pad,ty); BIDT(3);}
                if (tzset) {pad += 3; SETB(9+pad,tz); BIDT(6);}
                break;
            case 5:
                /* adding single atoms for cart. constraints */
                pad = 0; p1 = NXYZ; 
                *(b+0) = 1.0; *(b+1) = 0.0; *(b+2) = 0.0; 
                BIDX(p1); 
                break;
            case 6:
                /* adding single atoms for cart. constraints */
                pad = 0; p1 = NXYZ; 
                *(b+0) = 0.0; *(b+1) = 1.0; *(b+2) = 0.0; 
                BIDX(p1); 
                break;
            case 7:
                /* adding single atoms for cart. constraints */
                pad = 0; p1 = NXYZ; 
                *(b+0) = 0.0; *(b+1) = 0.0; *(b+2) = 1.0; 
                BIDX(p1); 
                break;
            default:
                return 2;  /* unknown internal coordinate type */
        }
        n = 3*atoms_per_internal[type]+pad;
        if (sort) bsort2(b, j, n); /* ensure that the column index is increasing */
        b += n;
        nb += n;
        nint++;
    }
    return 1;  /* this point should not be reached! */

#undef NXYZ
#undef BIDX
#undef SETVEC
#undef SETB
}

/* }}} */
THCTK_PRIVATE
int Bmatrix_pbc2(int nx, const vtype *x, const vtype *h, const vtype *hi, \
        const int *cint, vtype *b, int *jb, int *ib, int sort) {

    int type, nint = 0, n, *j;
    int p1, p2, p3, p4, nb = 0, pad = 0;
    int i1, i2, i3, i4;
    vec dsum, d1, d2, d3, d4, f1, f2, f3, f4, tmp, zero, tx, ty, tz;
    zero[0]=0.0; zero[1]=0.0, zero[2]=0.0;


#define NXYZ    (3*((*cint++)-1))
#define BIDX(p) *jb++ = p%nx; *jb++ = (++p)%nx; *jb++ = (++p)%nx;

#define SETVEC(vec1, num) vec1[0]= (*(b+num+0)); \
                          vec1[1]= (*(b+num+1)); \
                          vec1[2]= (*(b+num+2));
#define SETB(num, vec1)   (*(b+num+0)) = vec1[0]; \
                          (*(b+num+1)) = vec1[1]; \
                          (*(b+num+2)) = vec1[2]; 
#define SETBH(num, vec1)  (*(b+num+0)) = vec1[0]*(*(h+0))+vec1[1]*(*(h+1))+vec1[2]*(*(h+2)); \
                          (*(b+num+1)) = vec1[0]*(*(h+3))+vec1[1]*(*(h+4))+vec1[2]*(*(h+5)); \
                          (*(b+num+2)) = vec1[0]*(*(h+6))+vec1[1]*(*(h+7))+vec1[2]*(*(h+8)); 

#define BIDT(num) *jb++ = nx+num; *jb++ = nx+num+1; *jb++ = nx+num+2;

#define CALCF(pos, vec2)   vec2[0] = (*(x+pos+0))*(*(hi+0))+(*(x+pos+1))*(*(hi+1))+(*(x+pos+2))*(*(hi+2)); \
                           vec2[1] = (*(x+pos+0))*(*(hi+3))+(*(x+pos+1))*(*(hi+4))+(*(x+pos+2))*(*(hi+5)); \
                           vec2[2] = (*(x+pos+0))*(*(hi+6))+(*(x+pos+1))*(*(hi+7))+(*(x+pos+2))*(*(hi+8)); 

#define AXPY2(a,vec1,vec2) vec2[0] = a*vec1[0]; \
                           vec2[1] = a*vec1[1]; \
                           vec2[2] = a*vec1[2];
    /* for atoms that lie outside the cell we need to add components 
     * to the back-folded indices and also to the lattice vectors  */

    while (1) {
        type = *cint++;
        ib[nint] = nb;      /* nint iterates over the rows of B */
        j = jb;
        if (type <= 0) {
            return 0;
        }
        switch (type) {
            case 1:
                pad = 0;
                p1 = NXYZ; p2 = NXYZ;
                i1=p1; i2=p2;
                stretch(x+p1, x+p2, d1, d2);
                CALCF(p1,f1); CALCF(p2,f2);
                // atoms are the same in different cells 
                // so we only set one
                if (i1%nx==i2%nx) {
                    SETBH(0,zero); 
                    // pass on index of smallest number
                    BIDX(p1);
                    pad -= 3;
                }
                else {
                    SETBH(0,d1);SETBH(3,d2); 
                    BIDX(p1); BIDX(p2);
                }
                // set unit cell values
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                AXPY2(f1[0],d1,tmp);VADD(tx,tx,tmp);
                AXPY2(f1[1],d1,tmp);VADD(ty,ty,tmp);
                AXPY2(f1[2],d1,tmp);VADD(tz,tz,tmp);
                AXPY2(f2[0],d2,tmp);VADD(tx,tx,tmp);
                AXPY2(f2[1],d2,tmp);VADD(ty,ty,tmp);
                AXPY2(f2[2],d2,tmp);VADD(tz,tz,tmp);
                pad += 3; SETB(3+pad,tx); BIDT(0);
                pad += 3; SETB(3+pad,ty); BIDT(3);
                pad += 3; SETB(3+pad,tz); BIDT(6);
                break;
            case 2: // bending angles
                pad = 0;
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ;
                i1=p1; i2=p2; i3=p3;
                bend(x+p1, x+p2, x+p3, d1, d2, d3);
                CALCF(p1,f1); CALCF(p2,f2); CALCF(p3,f3);
                if (i1%nx==i2%nx && i2%nx==i3%nx) {
                    SETBH(0,zero);
                    //pass on smallest index
                    BIDX(p1);
                    pad -= 6;
                }
                else if (i1%nx==i2%nx || i1%nx==i3%nx || i2%nx==i3%nx ) {
                    if (i1%nx==i2%nx) {
                        VADD(dsum,d1,d2);
                        SETBH(0,dsum); SETBH(3,d3);
                        BIDX(p1); BIDX(p3);
                        pad -= 3;
                    }
                    else if (i2%nx==i3%nx) {
                        VADD(dsum,d2,d3);
                        SETBH(0,d1); SETBH(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 3;
                    }
                    else {
                        VADD(dsum,d1,d3);
                        SETBH(0,dsum); SETBH(3,d2);
                        BIDX(p1); BIDX(p2);
                        pad -= 3;
                    }
                }
                else { //default, no duplicate atoms
                    SETBH(0,d1); SETBH(3,d2); SETBH(6,d3);
                    BIDX(p1); BIDX(p2); BIDX(p3);
                }
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                AXPY2(f1[0],d1,tmp);VADD(tx,tx,tmp);
                AXPY2(f1[1],d1,tmp);VADD(ty,ty,tmp);
                AXPY2(f1[2],d1,tmp);VADD(tz,tz,tmp);
                AXPY2(f2[0],d2,tmp);VADD(tx,tx,tmp);
                AXPY2(f2[1],d2,tmp);VADD(ty,ty,tmp);
                AXPY2(f2[2],d2,tmp);VADD(tz,tz,tmp);
                AXPY2(f3[0],d3,tmp);VADD(tx,tx,tmp);
                AXPY2(f3[1],d3,tmp);VADD(ty,ty,tmp);
                AXPY2(f3[2],d3,tmp);VADD(tz,tz,tmp);
                pad += 3; SETB(6+pad,tx); BIDT(0);
                pad += 3; SETB(6+pad,ty); BIDT(3);
                pad += 3; SETB(6+pad,tz); BIDT(6);
                break;
            case 3: // torsions
                pad = 0;
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ; p4 = NXYZ;
                i1=p1; i2=p2;i3=p3;i4=p4;
                torsion(x+p1, x+p2, x+p3, x+p4, d1, d2, d3, d4);
                CALCF(p1,f1); CALCF(p2,f2); CALCF(p3,f3); CALCF(p4,f4);
                if (i1%nx==i2%nx && i2%nx==i3%nx && i3%nx==i4%nx) {
                    // all indices are the same
                    SETBH(0,zero);
                    //pass on smallest index
                    BIDX(p1);
                    pad -= 9;
                }
                else if ( (i1%nx==i2%nx && i2%nx==i3%nx) || \
                          (i1%nx==i2%nx && i2%nx==i4%nx) || \
                          (i2%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i2%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i2%nx==i4%nx) || \
                          (i1%nx==i4%nx && i2%nx==i3%nx) \
                          ) {
                    // three indices are the same
                    if (i1%nx==i2%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d3);
                        SETBH(0,dsum); SETBH(3,d4);
                        BIDX(p1); BIDX(p4);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d4);
                        SETBH(0,dsum); SETBH(3,d3);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i2%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d2,d3); VADD(dsum,dsum,d4);
                        SETBH(3,d1); SETBH(0,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d3); VADD(dsum,dsum,d4);
                        SETBH(0,dsum); SETBH(3,d2);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d2);
                        SETBH(0,dsum); 
                        VADD(dsum,d3,d4);
                        SETBH(3,dsum);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d3);
                        SETBH(0,dsum); 
                        VADD(dsum,d2,d4);
                        SETBH(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else  { //(i1%nx==i4%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d4);
                        SETBH(0,dsum); 
                        VADD(dsum,d2,d3);
                        SETBH(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                }
                else if (i1%nx==i2%nx || i1%nx==i3%nx || i1%nx==i4%nx || \
                         i2%nx==i3%nx || i2%nx==i4%nx || i3%nx==i4%nx ) {
                    // two indices are the same
                    if (i1%nx==i2%nx) {
                        VADD(dsum,d1,d2);
                        SETBH(0,dsum); SETBH(3,d3); SETBH(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i3%nx) {
                        VADD(dsum,d1,d3);
                        SETBH(0,dsum); SETBH(3,d2); SETBH(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i4%nx) {
                        VADD(dsum,d1,d4);
                        SETBH(0,dsum); SETBH(3,d2); SETBH(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else if (i2%nx==i3%nx) {
                        VADD(dsum,d2,d3);
                        SETBH(0,d1); SETBH(3,dsum); SETBH(6,d4);
                        BIDX(p1); BIDX(p2); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i2%nx==i4%nx) {
                        VADD(dsum,d2,d4);
                        SETBH(0,d1); SETBH(3,dsum); SETBH(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else {
                        VADD(dsum,d3,d4);
                        SETBH(0,d1); SETBH(3,d2); SETBH(6,dsum);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                }
                else { //default, no duplicate atoms
                    // no indices are the same
                    SETBH(0,d1); SETBH(3,d2); SETBH(6,d3); SETBH(9,d4);
                    BIDX(p1); BIDX(p2); BIDX(p3); BIDX(p4);
                }
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                AXPY2(f1[0],d1,tmp);VADD(tx,tx,tmp);
                AXPY2(f1[1],d1,tmp);VADD(ty,ty,tmp);
                AXPY2(f1[2],d1,tmp);VADD(tz,tz,tmp);
                AXPY2(f2[0],d2,tmp);VADD(tx,tx,tmp);
                AXPY2(f2[1],d2,tmp);VADD(ty,ty,tmp);
                AXPY2(f2[2],d2,tmp);VADD(tz,tz,tmp);
                AXPY2(f3[0],d3,tmp);VADD(tx,tx,tmp);
                AXPY2(f3[1],d3,tmp);VADD(ty,ty,tmp);
                AXPY2(f3[2],d3,tmp);VADD(tz,tz,tmp);
                AXPY2(f4[0],d4,tmp);VADD(tx,tx,tmp);
                AXPY2(f4[1],d4,tmp);VADD(ty,ty,tmp);
                AXPY2(f4[2],d4,tmp);VADD(tz,tz,tmp);
                pad += 3; SETB(9+pad,tx); BIDT(0);
                pad += 3; SETB(9+pad,ty); BIDT(3);
                pad += 3; SETB(9+pad,tz); BIDT(6);
                break;
            case 4:
                pad = 0;
                p1 = NXYZ; p2 = NXYZ; p3 = NXYZ; p4 = NXYZ;
                i1=p1;i2=p2;i3=p3;i4=p4;
                out_of_plane(x+p1, x+p2, x+p3, x+p4, d1, d2, d3, d4);
                CALCF(p1,f1); CALCF(p2,f2); CALCF(p3,f3); CALCF(p4,f4);
                if (i1%nx==i2%nx && i2%nx==i3%nx && i3%nx==i4%nx) {
                    // all indices are the same
                    SETBH(0,zero);
                    //pass on smallest index
                    BIDX(p1);
                    pad -= 9;
                }
                else if ( (i1%nx==i2%nx && i2%nx==i3%nx) || \
                          (i1%nx==i2%nx && i2%nx==i4%nx) || \
                          (i2%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i3%nx==i4%nx) || \
                          (i1%nx==i2%nx && i3%nx==i4%nx) || \
                          (i1%nx==i3%nx && i2%nx==i4%nx) || \
                          (i1%nx==i4%nx && i2%nx==i3%nx) \
                          ) {
                    // three indices are the same
                    if (i1%nx==i2%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d3);
                        SETBH(0,dsum); SETBH(3,d4);
                        BIDX(p1); BIDX(p4);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d2); VADD(dsum,dsum,d4);
                        SETBH(0,dsum); SETBH(3,d3);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i2%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d2,d3); VADD(dsum,dsum,d4);
                        SETBH(3,d1); SETBH(0,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d3); VADD(dsum,dsum,d4);
                        SETBH(0,dsum); SETBH(3,d2);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else if (i1%nx==i2%nx && i3%nx==i4%nx) {
                        VADD(dsum,d1,d2);
                        SETBH(0,dsum); 
                        VADD(dsum,d3,d4);
                        SETBH(3,dsum);
                        BIDX(p1); BIDX(p3);
                        pad -= 6;
                    }
                    else if (i1%nx==i3%nx && i2%nx==i4%nx) {
                        VADD(dsum,d1,d3);
                        SETBH(0,dsum); 
                        VADD(dsum,d2,d4);
                        SETBH(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                    else { //(i1%nx==i4%nx && i2%nx==i3%nx) {
                        VADD(dsum,d1,d4);
                        SETBH(0,dsum); 
                        VADD(dsum,d2,d3);
                        SETBH(3,dsum);
                        BIDX(p1); BIDX(p2);
                        pad -= 6;
                    }
                }
                else if (i1%nx==i2%nx || i1%nx==i3%nx || i1%nx==i4%nx || \
                         i2%nx==i3%nx || i2%nx==i4%nx || i3%nx==i4%nx ) {
                    // two indices are the same
                    if (i1%nx==i2%nx) {
                        VADD(dsum,d1,d2);
                        SETBH(0,dsum); SETBH(3,d3); SETBH(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i3%nx) {
                        VADD(dsum,d1,d3);
                        SETBH(0,dsum); SETBH(3,d2); SETBH(6,d4);
                        BIDX(p1); BIDX(p3); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i1%nx==i4%nx) {
                        VADD(dsum,d1,d4);
                        SETBH(0,dsum); SETBH(3,d2); SETBH(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else if (i2%nx==i3%nx) {
                        VADD(dsum,d2,d3);
                        SETBH(0,d1); SETBH(3,dsum); SETBH(6,d4);
                        BIDX(p1); BIDX(p2); BIDX(p4);
                        pad -= 3;
                    }
                    else if (i2%nx==i4%nx) {
                        VADD(dsum,d2,d4);
                        SETBH(0,d1); SETBH(3,dsum); SETBH(6,d3);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                    else {
                        VADD(dsum,d3,d4);
                        SETBH(0,d1); SETBH(3,d2); SETBH(6,dsum);
                        BIDX(p1); BIDX(p2); BIDX(p3);
                        pad -= 3;
                    }
                }
                else { //default, no duplicate atoms
                    // no indices are the same
                    SETBH(0,d1); SETBH(3,d2); SETBH(6,d3); SETBH(9,d4);
                    BIDX(p1); BIDX(p2); BIDX(p3); BIDX(p4);
                }
                VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);
                AXPY2(f1[0],d1,tmp);VADD(tx,tx,tmp);
                AXPY2(f1[1],d1,tmp);VADD(ty,ty,tmp);
                AXPY2(f1[2],d1,tmp);VADD(tz,tz,tmp);
                AXPY2(f2[0],d2,tmp);VADD(tx,tx,tmp);
                AXPY2(f2[1],d2,tmp);VADD(ty,ty,tmp);
                AXPY2(f2[2],d2,tmp);VADD(tz,tz,tmp);
                AXPY2(f3[0],d3,tmp);VADD(tx,tx,tmp);
                AXPY2(f3[1],d3,tmp);VADD(ty,ty,tmp);
                AXPY2(f3[2],d3,tmp);VADD(tz,tz,tmp);
                AXPY2(f4[0],d4,tmp);VADD(tx,tx,tmp);
                AXPY2(f4[1],d4,tmp);VADD(ty,ty,tmp);
                AXPY2(f4[2],d4,tmp);VADD(tz,tz,tmp);
                pad += 3; SETB(9+pad,tx); BIDT(0);
                pad += 3; SETB(9+pad,ty); BIDT(3);
                pad += 3; SETB(9+pad,tz); BIDT(6);
                break;
            case 5:
                /* adding single atoms for cart. constraints */
                pad = 0; p1 = NXYZ; 
                CALCF(p1,f1);
                tmp[0]=(*(h+0)); tmp[1]=(*(h+1)); tmp[2]=(*(h+2));
                SETBH(0,tmp);
                BIDX(p1); 
                /*VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);*/
                /*AXPY2(f1[0],d1,tmp);VADD(tx,tx,tmp);*/
                /*AXPY2(f1[1],d1,tmp);VADD(ty,ty,tmp);*/
                /*AXPY2(f1[2],d1,tmp);VADD(tz,tz,tmp);*/
                /*pad += 3; SETB(pad,tx); BIDT(0);*/
                /*pad += 3; SETB(pad,ty); BIDT(3);*/
                /*pad += 3; SETB(pad,tz); BIDT(6);*/
                break;
            case 6:
                /* adding single atoms for cart. constraints */
                pad = 0; p1 = NXYZ; 
                CALCF(p1,f1);
                tmp[0]=(*(h+3)); tmp[1]=(*(h+4)); tmp[2]=(*(h+5)); 
                SETBH(0,tmp);
                BIDX(p1); 
                /*VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);*/
                /*AXPY2(f1[0],d1,tmp);VADD(tx,tx,tmp);*/
                /*AXPY2(f1[1],d1,tmp);VADD(ty,ty,tmp);*/
                /*AXPY2(f1[2],d1,tmp);VADD(tz,tz,tmp);*/
                /*pad += 3; SETB(pad,tx); BIDT(0);*/
                /*pad += 3; SETB(pad,ty); BIDT(3);*/
                /*pad += 3; SETB(pad,tz); BIDT(6);*/
                break;
            case 7:
                /* adding single atoms for cart. constraints */
                pad = 0; p1 = NXYZ; 
                CALCF(p1,f1);
                tmp[0]=(*(h+6)); tmp[1]=(*(h+7)); tmp[2]=(*(h+8)); 
                SETBH(0,tmp);
                BIDX(p1); 
                /*VCP(tx,zero); VCP(ty,zero); VCP(tz,zero);*/
                /*AXPY2(f1[0],d1,tmp);VADD(tx,tx,tmp);*/
                /*AXPY2(f1[1],d1,tmp);VADD(ty,ty,tmp);*/
                /*AXPY2(f1[2],d1,tmp);VADD(tz,tz,tmp);*/
                /*pad += 3; SETB(pad,tx); BIDT(0);*/
                /*pad += 3; SETB(pad,ty); BIDT(3);*/
                /*pad += 3; SETB(pad,tz); BIDT(6);*/
                break;
            default:
                return 2;  /* unknown internal coordinate type */
        }
        n = 3*atoms_per_internal[type]+pad;
        if (sort) bsort2(b, j, n); /* ensure that the column index is increasing */
        b += n;
        nb += n;
        nint++;
    }
    return 1;  /* this point should not be reached! */

#undef NXYZ
#undef BIDX
#undef SETVEC
#undef SETB
#undef SETBH
#undef BIDT
#undef CALCF
#undef AXPY2
}

/* }}} */

/* {{{ Btrans_p */

THCTK_PRIVATE
int Btrans_p(int nnz, int m, int n, int update, const int *p,
    const vtype *b, const int *jb, const int *ib, vtype *a, int *ja, int* ia) {

    /* A = (P B)^T , with column permutation P */

    int i;

    if (! update) { /* we need to compute the structure of A */
        for (i=0; i<n+1; i++) ia[i] = 0;
        for (i=0; i<nnz; i++) ia[p[jb[i]]+1]++;    /* count column elements of B */
        for (i=0; i<n; i++) ia[i+1] += ia[i];   /* compute A offsets from counts */
    }

    for (i=0; i<m; i++) {   /* perform the transposition */
        register int k;
        for (k=ib[i]; k<ib[i+1]; k++) {
            register int j = ia[p[jb[k]]]++;
            a[j] = b[k];
            ja[j] = i;
        }
    }

    for (i=n; i>0; i--) ia[i] = ia[i-1];    /* shift offsets back */
    ia[0] = 0;

    return 0;
}

/* }}} */

/* {{{ Btrans*/

THCTK_PRIVATE
int Btrans(int nnz, int m, int n, int update,
    const vtype *b, const int *jb, const int *ib, vtype *a, int *ja, int* ia) {

    /* A = B^T */

    int i;

    if (! update) { /* we need to compute the structure of A */
        for (i=0; i<n+1; i++) ia[i] = 0;
        for (i=0; i<nnz; i++) ia[jb[i]+1]++;    /* count column elements of B */
        for (i=0; i<n; i++) ia[i+1] += ia[i];   /* compute A offsets from counts */
    }

    for (i=0; i<m; i++) {   /* perform the transposition */
        register int k;
        for (k=ib[i]; k<ib[i+1]; k++) {
            register int j = ia[jb[k]]++;
            a[j] = b[k];
            ja[j] = i;
        }
    }

    for (i=n; i>0; i--) ia[i] = ia[i-1];    /* shift offsets back */
    ia[0] = 0;

    return 0;
}

/* }}} */

/* {{{ BxBt_d */

THCTK_PRIVATE
int BxBt_d(int n, const vtype *b, const int *jb, const int *ib, int nnz,
    vtype *a, const int *ja, const int* ia, vtype *diag) {

    /*
     * This routine computes the symmetric (nxn) normal matrix A corresponding
     * to a sparse rectangular (nxm) matrix B given in CSR format with column
     * indices in increasing order. Formally:
     *          A = B B^T
     * The strict upper triangle of A is returned in ordered CSR format. The
     * diagonal of a is stored in diag .
     */

    int i, j, k, ip, ibeg=0, iend=0, jp, jend;
    vtype sum;

    /* compute the diagonal elements */
    for (i=0; i<n; i++) {
        sum = 0;
        for (k=ib[i]; k<ib[i+1]; k++) sum += b[k]*b[k];
        diag[i] = sum;
    }

    /* we know the structure of A and only update the numerical values */
    for (k=0, i=-1; k<nnz; k++) { /* iterate over the elements of A */
        j = ja[k];
        jend = ib[j+1];
        while (k >= *ia) { /* did we reach a new row? */
            ia++;
            i++;
            ibeg = ib[i];
            iend = ib[i+1];
        }
        ip = ibeg;
        jp = ib[j];
        sum = 0;
        while (ip < iend && jp < jend) {
            /* compute the intersection of column indices */
            if      (jb[ip] > jb[jp])   jp++;
            else if (jb[ip] < jb[jp])   ip++;
            else                        sum += b[ip++] * b[jp++];
        }
        *a++ = sum;
    }

    return 0;
}
        
/* }}} */

/* {{{ BxBt */

THCTK_PRIVATE
int BxBt(int n, const vtype *b, const int *jb, const int *ib,
    vtype *a, int *ja, int* ia, int *nnz, int flag) {

    /*
     * This routine computes the symmetric (nxn) normal matrix A corresponding
     * to a sparse rectangular (nxm) matrix B given in CSR format with column
     * indices in increasing order. Formally:
     *          A = B B^T
     * The upper triangle of A is returned in ordered CSR format.
     */

    int i, j, ip, ibeg=0, iend=0, jp, jend;
    vtype sum;

    if (flag == 0) {
        /* we know the structure of A and only update the numerical values */
        int k;
        for (k=0, i=-1; k < *nnz; k++) { /* iterate over the elements of A */
            j = ja[k];
            while (k >= *ia) { /* did we reach a new row? */
                ia++;
                i++;
                ibeg = ib[i];
                iend = ib[i+1];
            }
            ip = ibeg;
            sum = 0; /* accumulate the dot product of rows i and j */
            if (i == j) {   /* diagonal element */
                while (ip < iend) {
                    register vtype t = b[ip++];
                    sum += t*t;
                }
            } else {
                jp = ib[j];
                jend = ib[j+1];
                while (ip < iend && jp < jend) {
                    /* compute the intersection of column indices */
                    if      (jb[ip] > jb[jp])   jp++;
                    else if (jb[ip] < jb[jp])   ip++;
                    else                        sum += b[ip++] * b[jp++];
                }
            }
            *a++ = sum;
        }
    } else {
        /* for (flag == 1) only nnz and the row indices are determined */
        *nnz = 0;
        for (i=0; i<n; i++) {
            ibeg = ib[i];
            iend = ib[i+1];
            *ia++ = *nnz;    /* start a new row */

            ip = ibeg;       /* compute the diagonal element */
            sum = 0;
            while (ip < iend) {
                register vtype t = b[ip++];
                sum += t*t;
            }
            if (sum) {
                if (flag != 1) {
                    *a++ = sum;
                    *ja++ = i;
                }
                (*nnz)++;
            }

            for (j=i+1; j<n; j++) {   /* compute the strict upper triangle */
                jend = ib[j+1];
                ip = ibeg;
                jp = ib[j];
                sum = 0; /* accumulate the dot product of rows i and j */
                while (ip < iend && jp < jend) {
                    /* compute the intersection of column indices */
                    if      (jb[ip] > jb[jp])   jp++;
                    else if (jb[ip] < jb[jp])   ip++;
                    else                        sum += b[ip++] * b[jp++];
                }
                if (sum) {
                    if (flag != 1) {
                        *a++ = sum;
                        *ja++ = j;
                    }
                    (*nnz)++;
                }
            }
        }
        *ia = *nnz;
    }

    return 0;
}

/* }}} */

/* {{{ BtxB */

THCTK_PRIVATE
int BtxB(int n, const vtype *b, const int *jb, const int *ib,
    vtype *a, int *ja, int* ia, int *nnz, int flag) {

    /*
     * This routine computes the symmetric (mxm) normal matrix G corresponding
     * to a sparse rectangular (nxm) matrix B given in CSR format with column
     * indices in increasing order. Formally:
     *          A = B^T B
     * The upper triangle of G is returned in ordered CSR format.
     */

    int i, j, ip, ibeg=0, iend=0, jp, jend;
    vtype sum;

    if (flag == 0) {
        /* we know the structure of G and only update the numerical values */
        int k;
        for (k=0, i=-1; k < *nnz; k++) { /* iterate over the elements of G */
            j = ja[k];
            while (k >= *ia) { /* did we reach a new row? */
                ia++;
                i++;
                ibeg = ib[i];
                iend = ib[i+1];
            }
            ip = ibeg;
            sum = 0; /* accumulate the dot product of rows i and j */
            if (i == j) {   /* diagonal element */
                while (ip < iend) {
                    register vtype t = b[ip++];
                    sum += t*t;
                }
            } else {
                jp = ib[j];
                jend = ib[j+1];
                while (ip < iend && jp < jend) {
                    /* compute the intersection of column indices */
                    if      (jb[ip] > jb[jp])   jp++;
                    else if (jb[ip] < jb[jp])   ip++;
                    else                        sum += b[ip++] * b[jp++];
                }
            }
            *a++ = sum;
        }
    } else {
        /* for (flag == 1) only nnz and the row indices are determined */
        *nnz = 0;
        for (i=0; i<n; i++) {
            ibeg = ib[i];
            iend = ib[i+1];
            *ia++ = *nnz;    /* start a new row */

            ip = ibeg;       /* compute the diagonal element */
            sum = 0;
            while (ip < iend) {
                register vtype t = b[ip++];
                sum += t*t;
            }
            if (sum) {
                if (flag != 1) {
                    *a++ = sum;
                    *ja++ = i;
                }
                (*nnz)++;
            }

            for (j=i+1; j<n; j++) {   /* compute the strict upper triangle */
                jend = ib[j+1];
                ip = ibeg;
                jp = ib[j];
                sum = 0; /* accumulate the dot product of rows i and j */
                while (ip < iend && jp < jend) {
                    /* compute the intersection of column indices */
                    if      (jb[ip] > jb[jp])   jp++;
                    else if (jb[ip] < jb[jp])   ip++;
                    else                        sum += b[ip++] * b[jp++];
                }
                if (sum) {
                    if (flag != 1) {
                        *a++ = sum;
                        *ja++ = j;
                    }
                    (*nnz)++;
                }
            }
        }
        *ia = *nnz;
    }

    return 0;
}

/* }}} */
/* }}} */

/* {{{ python interface */
#ifdef THCTK_INTERFACE


THCTKDOC(_intcrd, Bmatrix_nnz) = "nnz, nint = Bmatrix_nnz(cint)\n";

THCTKFUN(_intcrd, Bmatrix_nnz)
{

    PyObject *input;
    PyArrayObject *cint;
    int nnz, nint;

    if (!PyArg_ParseTuple(args, "O", &input)) return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;

    if (Bmatrix_nnz((int *) cint->data, &nint, &nnz)) return NULL;

    return Py_BuildValue("ii", nnz, nint);

}

THCTKDOC(_intcrd, Bmatrix_pbc_nnz) = "nnz, nint = Bmatrix_pbc_nnz(nx, cint)\n";

THCTKFUN(_intcrd, Bmatrix_pbc_nnz)
{

    PyObject *input;
    PyArrayObject *cint;
    int nx, nnz, nint;

    if (!PyArg_ParseTuple(args, "iO", &nx, &input)) return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;

    if (Bmatrix_pbc_nnz(nx, (int *) cint->data, &nint, &nnz)) return NULL;

    return Py_BuildValue("ii", nnz, nint);

}


THCTKDOC(_intcrd, Bmatrix_pbc2_nnz) = "nnz, nint = Bmatrix_pbc2_nnz(nx, cint)\n";

THCTKFUN(_intcrd, Bmatrix_pbc2_nnz)
{

    PyObject *input;
    PyArrayObject *cint;
    int nx, nnz, nint;

    if (!PyArg_ParseTuple(args, "iO", &nx, &input)) return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;

    if (Bmatrix_pbc2_nnz(nx, (int *) cint->data, &nint, &nnz)) return NULL;

    return Py_BuildValue("ii", nnz, nint);

}


THCTKDOC(_intcrd, internals) = "c = internals(x, cint, c)\n";

THCTKFUN(_intcrd, internals)
{

    PyObject *input;
    PyArrayObject *x, *cint, *c;

    if (!PyArg_ParseTuple(args, "O!OO!", &PyArray_Type, &x,
        &input, &PyArray_Type, &c)) return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;


    if (internals((double *) x->data, (int *) cint->data,
                  (double *) c->data)) return NULL;

    Py_INCREF(c);
    return (PyObject *) c;

}

THCTKDOC(_intcrd, internals_pbc) = "c = internals_pbc(x, h, cint, c)\n";

THCTKFUN(_intcrd, internals_pbc)
{

    PyObject *input;
    PyArrayObject *x, *h, *cint, *c;

    if (!PyArg_ParseTuple(args, "O!O!OO!", &PyArray_Type, &x, &PyArray_Type, &h,
        &input, &PyArray_Type, &c)) return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;


    if (internals_pbc((double *) x->data, (double *) h->data, (int *) cint->data,
                  (double *) c->data)) return NULL;

    Py_INCREF(c);
    return (PyObject *) c;

}


THCTKDOC(_intcrd, Bmatrix) = "Bmatrix(x, cint, b, jb, ib, sort)";

THCTKFUN(_intcrd, Bmatrix)
{

    PyObject *input;
    PyArrayObject *x, *cint, *b, *jb, *ib;
    int sort;

    if (!PyArg_ParseTuple(args, "O!OO!O!O!i", &PyArray_Type, &x,
        &input, &PyArray_Type, &b, &PyArray_Type, &jb,
        &PyArray_Type, &ib, &sort))
        return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;

    if (Bmatrix((double *) x->data, (int *) cint->data,
                (double *) b->data, (int *) jb->data, (int *) ib->data, sort))
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}

THCTKDOC(_intcrd, Bmatrix_pbc) = "Bmatrix_pbc(nx, x, cint, b, jb, ib, sort)";

THCTKFUN(_intcrd, Bmatrix_pbc)
{

    PyObject *input;
    PyArrayObject *x, *cint, *b, *jb, *ib ;
    int sort, nx;

    if (!PyArg_ParseTuple(args, "iO!OO!O!O!i", &nx, &PyArray_Type, &x,
        &input, &PyArray_Type, &b, &PyArray_Type, &jb,
        &PyArray_Type, &ib, &sort))
        return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;

    if (Bmatrix_pbc(nx, (double *) x->data, (int *) cint->data,
                (double *) b->data, (int *) jb->data, (int *) ib->data,
                sort))
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}

THCTKDOC(_intcrd, Bmatrix_pbc2) = "Bmatrix_pbc2(nx, x, h, h2, cint, b, jb, ib, sort)";

THCTKFUN(_intcrd, Bmatrix_pbc2)
{

    PyObject *input;
    PyArrayObject *x, *h, *h2, *cint, *b, *jb, *ib ;
    int sort, nx;

    if (!PyArg_ParseTuple(args, "iO!OO!O!O!O!O!i", &nx, &PyArray_Type, &x,
        &input, &PyArray_Type, &h, &PyArray_Type, &h2, &PyArray_Type, &b, &PyArray_Type, &jb,
        &PyArray_Type, &ib, &sort))
        return NULL;

    cint = (PyArrayObject *)
        PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;

    if ( Bmatrix_pbc2(nx, (double *) x->data, (double *) h->data, (double *) h2->data, 
                (int *) cint->data, (double *) b->data, (int *) jb->data, (int *) ib->data,
                sort))
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}


THCTKDOC(_intcrd, Btrans_p) =
"Btrans_p(nnz, m, n, update, p, b, jb, ib, a, ja, ia)\n"
"    A = (P B)^T\n"
"where A is (nxm) and B is (mxn), both in CSR format. P is an\n"
"integer vector of length m representing a column permutation of B.\n";

THCTKFUN(_intcrd, Btrans_p)
{

    PyArrayObject *p, *b, *jb, *ib, *a, *ja, *ia;
    int m, n, nnz, update;

    if (!PyArg_ParseTuple(args, "iiiiO!O!O!O!O!O!O!",
        &nnz, &m, &n, &update, &PyArray_Type, &p,
        &PyArray_Type, &b, &PyArray_Type, &jb, &PyArray_Type, &ib,
        &PyArray_Type, &a, &PyArray_Type, &ja, &PyArray_Type, &ia))
        return NULL;

    if ( Btrans_p(nnz, m, n, update, (int *) p->data,
                  (double *) b->data, (int *) jb->data, (int *) ib->data,
                  (double *) a->data, (int *) ja->data, (int *) ia->data) )
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}

THCTKDOC(_intcrd, Btrans) =
"Btrans(nnz, m, n, update, b, jb, ib, a, ja, ia)\n"
"    A = B^T\n"
"where A is (nxm) and B is (mxn), both in CSR format";

THCTKFUN(_intcrd, Btrans)
{

    PyArrayObject *b, *jb, *ib, *a, *ja, *ia;
    int m, n, nnz, update;

    if (!PyArg_ParseTuple(args, "iiiiO!O!O!O!O!O!", &nnz, &m, &n, &update,
        &PyArray_Type, &b, &PyArray_Type, &jb, &PyArray_Type, &ib,
        &PyArray_Type, &a, &PyArray_Type, &ja, &PyArray_Type, &ia))
        return NULL;

    if ( Btrans(nnz, m, n, update,
                (double *) b->data, (int *) jb->data, (int *) ib->data,
                (double *) a->data, (int *) ja->data, (int *) ia->data) )
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}


THCTKDOC(_intcrd, conn2crd) =
"conn2crd(natom, diag, cj, ci, aj, ai)";

THCTKFUN(_intcrd, conn2crd)
{

    PyArrayObject *cj, *ci, *aj, *ai;
    int natom, diag;

    if (!PyArg_ParseTuple(args, "iiO!O!O!O!", &natom, &diag, &PyArray_Type, &cj,
        &PyArray_Type, &ci, &PyArray_Type, &aj, &PyArray_Type, &ai))
        return NULL;

    if ( conn2crd(natom, diag, (int *) cj->data, (int *) ci->data,
            (int *) aj->data, (int *) ai->data) ) return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}


THCTKDOC(_intcrd, conn2crd_p) =
"ai, aj = conn2crd_p(natom, diag, cj, ci, p, sort)";

THCTKFUN(_intcrd, conn2crd_p)
{

    PyArrayObject *p, *ci, *cj;
    PyObject *ai, *aj;
    int natom, errcode, n, nx, diag, sort;
    struct ijarray *aij;

    if (!PyArg_ParseTuple(args, "iiO!O!O!i", &natom, &diag, &PyArray_Type, &cj,
        &PyArray_Type, &ci, &PyArray_Type, &p, &sort))
        return NULL;

    if (! (aij = conn2crd_p(natom, diag, (int *) cj->data, (int *) ci->data,
        (int *) p->data, sort, &errcode)) )
        return NULL;

    nx = 3*natom;
    if (p->dimensions[0] != nx) return NULL;

    n = nx + 1;
    if (! (ai = PyArray_FromDimsAndData(1, &n, PyArray_INT, (char *) aij->i)) )
        return NULL;

    n = aij->i[nx];
    if (! (aj = PyArray_FromDimsAndData(1, &n, PyArray_INT, (char *) aij->j)) )
        return NULL;

    return Py_BuildValue("OO", aj, ai);

}

THCTKDOC(_intcrd, symbolicAc) =
"cj, ci = symbolicAc(cint, natom)\n"
"This routine returns in cj and ci the connectivity of atoms as given\n"
"by the internal coordinates cint\n"
"The non-zero structure of the matrix A_c = B^T B in CSR format,\n"
"where B is the Wilson matrix, is obtained from cj, ci by introducing\n"
"a 3x3 block for the x, y, z coordinates of each connection.\n"
;

THCTKFUN(_intcrd, symbolicAc)
{

    PyArrayObject *cint;
    PyObject *ci, *cj, *input;
    int natom, errcode, n;
    struct ijarray *cij;

    if (!PyArg_ParseTuple(args, "Oi", &input, &natom))
        return NULL;

    cint = (PyArrayObject *)
        /* Deprecated */
        /* PyArray_ContiguousFromObject(input, PyArray_INT, 1, 1); */
        PyArray_ContiguousFromAny(input, PyArray_INT, 1, 1);

    if (cint == NULL) return NULL;

    if (! (cij = symbolicAc((int *) cint->data, natom, &errcode)) ) {
        printf("done\n");
        return NULL;
    }

    n = natom + 1;
    if (! (ci = PyArray_FromDimsAndData(1, &n, PyArray_INT, (char *) cij->i)) )
        return NULL;

    n = cij->i[natom];
    if (! (cj = PyArray_FromDimsAndData(1, &n, PyArray_INT, (char *) cij->j)) )
        return NULL;

    return Py_BuildValue("OO", cj, ci);

}


THCTKDOC(_intcrd, BxBt_d) =
"BxBt_d(n, b, jb, ib, nnz, a, ja, ia, diag)\n\n"
"This routine computes the symmetric (nxn) normal matrix A corresponding\n"
"to a sparse rectangular (nxm) matrix B given in CSR format with column\n"
"indices in increasing order. Formally:\n"
"         A = B B^T\n"
"The strict upper triangle of A is returned in ordered CSR format. The\n"
"diagonal of a is stored in diag .\n";

THCTKFUN(_intcrd, BxBt_d)
{

    PyArrayObject *b, *jb, *ib, *a, *ja, *ia, *diag;
    int n, nnz;

    if (!PyArg_ParseTuple(args, "iO!O!O!iO!O!O!O!", &n,
        &PyArray_Type, &b, &PyArray_Type, &jb, &PyArray_Type, &ib, &nnz,
        &PyArray_Type, &a, &PyArray_Type, &ja, &PyArray_Type, &ia,
        &PyArray_Type, &diag))
        return NULL;

    if ( BxBt_d(n, (double *) b->data, (int *) jb->data, (int *) ib->data, nnz,
        (double *) a->data, (int *) ja->data, (int *) ia->data,
        (double *) diag->data) ) return NULL;

    Py_INCREF(Py_None);
    return Py_None;

}


THCTKDOC(_intcrd, BxBt) =
"BxBt(n, b, jb, ib, a, ja, ia, nnz, flag)\n\n"
"This routine computes the symmetric (nxn) normal matrix A corresponding\n"
"to a sparse rectangular (nxm) matrix B given in CSR format with column\n"
"indices in increasing order. Formally:\n"
"         A = B B^T\n"
"The upper triangle of A is returned in ordered CSR format.";

THCTKFUN(_intcrd, BxBt)
{

    PyArrayObject *b, *jb, *ib, *a, *ja, *ia;
    int n, nnz, flag;

    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!O!ii", &n,
        &PyArray_Type, &b, &PyArray_Type, &jb, &PyArray_Type, &ib,
        &PyArray_Type, &a, &PyArray_Type, &ja, &PyArray_Type, &ia,
        &nnz, &flag))
        return NULL;

    if ( BxBt(n, (double *) b->data, (int *) jb->data, (int *) ib->data,
                 (double *) a->data, (int *) ja->data, (int *) ia->data,
                 &nnz, flag) ) return NULL;

    return Py_BuildValue("i", nnz);

}

THCTKDOC(_intcrd, BtxB) =
"BtxB(n, b, jb, ib, a, ja, ia, nnz, flag)\n\n"
"This routine computes the symmetric (mxm) normal matrix G corresponding\n"
"to a sparse rectangular (nxm) matrix B given in CSR format with column\n"
"indices in increasing order. Formally:\n"
"         A = B^T B\n"
"The upper triangle of G is returned in ordered CSR format.";

THCTKFUN(_intcrd, BtxB)
{

    PyArrayObject *b, *jb, *ib, *a, *ja, *ia;
    int n, nnz, flag;

    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!O!ii", &n,
        &PyArray_Type, &b, &PyArray_Type, &jb, &PyArray_Type, &ib,
        &PyArray_Type, &a, &PyArray_Type, &ja, &PyArray_Type, &ia,
        &nnz, &flag))
        return NULL;

    if ( BtxB(n, (double *) b->data, (int *) jb->data, (int *) ib->data,
                 (double *) a->data, (int *) ja->data, (int *) ia->data,
                 &nnz, flag) ) return NULL;

    return Py_BuildValue("i", nnz);

}


THCTKDOC(_intcrd, dphi_mod_2pi) =
"dphi = dphi_mod_2pi(dphi, idx)\n\n"
"dphi ... Numerical array of type Float\n"
"idx  ... Numerical array of type Int or Long\n\n"
"The elements in dphi indicated by the indices in idx contain the difference\n"
"between two torsion angles phi, theta in the range (-pi, pi] that are only\n"
"determined modulo 2*pi. The return values are the corresponding minimum\n"
"absolute distances min(abs(phi - theta)).\n";

THCTKFUN(_intcrd, dphi_mod_2pi)
{

    PyArrayObject *dphi, *idx;
    int i;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &dphi,
        &PyArray_Type, &idx))
        return NULL;

    if (dphi->nd != 1 || dphi->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,
            "dphi array must be one-dimensional and of type float");
        return NULL;
    }

    if (idx->nd != 1 || (idx->descr->type_num != PyArray_LONG &&
        idx->descr->type_num != PyArray_INT)) {
        PyErr_SetString(PyExc_ValueError,
            "idx array must be one-dimensional and of type int");
        return NULL;
    }

    for (i=0; i<idx->dimensions[0]; i++) {
        int k = *(int *)(idx->data + i*idx->strides[0]);
        double *d = (double *)(dphi->data + k*dphi->strides[0]);
        if (fabs(*d) > M_PI) {
            if (fabs(*d + 2*M_PI) < M_PI) {
                *d += 2*M_PI;
            } else {
                *d -= 2*M_PI;
            }
        }
    }

    Py_INCREF(dphi);
    return (PyObject *) dphi;

}


/* module initialization */

static struct PyMethodDef _intcrd_methods[] = {
    THCTKDEF(_intcrd, dphi_mod_2pi)
    THCTKDEF(_intcrd, BxBt)
    THCTKDEF(_intcrd, BtxB)
    THCTKDEF(_intcrd, BxBt_d)
    THCTKDEF(_intcrd, Btrans)
    THCTKDEF(_intcrd, Btrans_p)
    THCTKDEF(_intcrd, conn2crd)
    THCTKDEF(_intcrd, conn2crd_p)
    THCTKDEF(_intcrd, symbolicAc)
    THCTKDEF(_intcrd, internals)
    THCTKDEF(_intcrd, internals_pbc)
    THCTKDEF(_intcrd, Bmatrix)
    THCTKDEF(_intcrd, Bmatrix_pbc)
    THCTKDEF(_intcrd, Bmatrix_pbc2)
    THCTKDEF(_intcrd, Bmatrix_nnz)
    THCTKDEF(_intcrd, Bmatrix_pbc_nnz)
    THCTKDEF(_intcrd, Bmatrix_pbc2_nnz)
    {NULL, NULL, 0, NULL}
};

static char _intcrd_module_documentation[] = "";

THCTKMOD(_intcrd)

#endif
/* }}} */
