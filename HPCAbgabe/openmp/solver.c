/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * 	   Fabian Schneider (f.schneider@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENACC
# include <openacc.h>
#endif

#ifdef CUDA
# include <cuda.h>
#endif

#include "solver.h"
#include "output.h"


/* ab <- a' * b */
void vectorDot(const floatType* a, const floatType* b, const int n, floatType* ab){
	int i;
	floatType temp;
	temp=0;
	#pragma omp parallel for reduction(+:temp) default(none) shared(a,b,n) private(i) 
	for(i=0; i<n; i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

/* y <- ax + y */
void axpy(const floatType a, const floatType* x, const int n, floatType* y){
	int i;
	#pragma omp parallel for default(none) shared(y,a,x,n) private(i)
	for(i=0; i<n; i++){
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
void xpay(const floatType* x, const floatType a, const int n, floatType* y){
	int i;
	#pragma omp parallel for default(none) shared(y,x,a,n) private(i)
	for(i=0; i<n; i++){
		y[i]=x[i]+a*y[i];
	}
}

/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
void matvec(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y){
	int i, j, k;
	floatType tmp;	
	#pragma omp parallel default(none) shared(y,length,data,x,indices,n) private(i,j,k,tmp)
	#pragma omp for
	for (i = 0; i < n; i++) {
		tmp = 0; 
		for (j = 0; j < length[i]; j++) {
			k = (i*maxNNZ) + j;
			tmp += data[k] * x[indices[k]];
		}
		y[i] = tmp;
	}
}

/* y <- C*x where C is diag(A)^{-1} for Jacobi Preconditioner */
void matvec_diag(const int n, const int nnz, const int maxNNZ, const floatType* data_diag, const floatType* x, floatType* y){
	int i;
	#pragma omp parallel default(none) shared(y, data_diag, x, n) private(i)
	#pragma omp for
	for(i = 0; i < n; i++){
		y[i] = data_diag[i] * x[i];
	}
}

/* nrm <- ||x||_2 */
void nrm2(const floatType* x, const int n, floatType* nrm){
	int i;
	floatType temp;
	temp = 0;
	#pragma omp parallel for default(none) shared(x) private(i) reduction(+:temp)
	for(i = 0; i<n; i++){
		temp+=(x[i]*x[i]);
	}
	*nrm=sqrt(temp);
}


/***************************************
 *         Conjugate Gradient          *
 *   This function will do the CG      *
 *  algorithm without preconditioning. *
 *    For optimiziation you must not   *
 *        change the algorithm.        *
 ***************************************
 r(0)    = b - Ax(0)
 p(0)    = r(0)
 rho(0)    =  <r(0),r(0)>                
 ***************************************
 for k=0,1,2,...,n-1
   q(k)      = A * p(k)                 
   dot_pq    = <p(k),q(k)>             
   alpha     = rho(k) / dot_pq
   x(k+1)    = x(k) + alpha*p(k)      
   r(k+1)    = r(k) - alpha*q(k)     
   check convergence ||r(k+1)||_2 < eps  
	 rho(k+1)  = <r(k+1), r(k+1)>         
   beta      = rho(k+1) / rho(k)
   p(k+1)    = r(k+1) + beta*p(k)      
***************************************/
void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* data_diag, const floatType* b, floatType* x, struct SolverConfig* sc){
	floatType* r, *p, *q, *h;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2;
	int iter;
 	double timeMatvec_s;
 	double timeMatvec=0;
	
	/* allocate memory */
	r = (floatType*)malloc(n * sizeof(floatType));
	p = (floatType*)malloc(n * sizeof(floatType));
	q = (floatType*)malloc(n * sizeof(floatType));
	h = (floatType*)malloc(n * sizeof(floatType)); // for Jacobi Preconditioner
	
	DBGMAT("Start matrix A = ", n, nnz, maxNNZ, data, indices, length)
	DBGVEC("b = ", b, n);
	DBGVEC("x = ", x, n);

	/* r(0)    = b - Ax(0) */
	timeMatvec_s = getWTime();
	matvec(n, nnz, maxNNZ, data, indices, length, x, r);
	timeMatvec += getWTime() - timeMatvec_s;
	xpay(b, -1.0, n, r);
	DBGVEC("r = b - Ax = ", r, n);
	
	/* h(0) = C * r(0) */
	matvec_diag(n, nnz, maxNNZ, data_diag, r, h);
	DBGVEC("h = C * r= ", h, n);

	/* Calculate initial residuum */
	nrm2(h, n, &bnrm2);
	bnrm2 = 1.0 /bnrm2;

	/* p(0)    = h(0) */
	memcpy(p, h, n*sizeof(floatType));
	DBGVEC("p = h = ", p, n);

	/* rho(0)    =  <r(0),h(0)> */
	vectorDot(r, h, n, &rho);
	printf("rho_0=%e\n", rho);

	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
	
		/* q(k)      = A * p(k) */
		timeMatvec_s = getWTime();
		matvec(n, nnz, maxNNZ, data, indices, length, p, q);
		timeMatvec += getWTime() - timeMatvec_s;
		DBGVEC("q = A * p= ", q, n);

		/* dot_pq    = <p(k),q(k)> */
		vectorDot(p, q, n, &dot_pq);
		DBGSCA("dot_pq = <p, q> = ", dot_pq);

		/* alpha     = rho(k) / dot_pq */
		alpha = rho / dot_pq;
		DBGSCA("alpha = rho / dot_pq = ", alpha);

		/* x(k+1)    = x(k) + alpha*p(k) */
		axpy(alpha, p, n, x);
		DBGVEC("x = x + alpha * p= ", x, n);

		/* r(k+1)    = r(k) - alpha*q(k) */
		axpy(-alpha, q, n, r);
		DBGVEC("r = r - alpha * q= ", r, n);

		/* h(k+1) = C * r(k+1) */
		matvec_diag(n, nnz, maxNNZ, data_diag, r, h);
		DBGVEC("h = C * r= ", h, n);

		rho_old = rho;
		DBGSCA("rho_old = rho = ", rho_old);


		/* rho(k+1)  = <r(k+1), h(k+1)> */
		vectorDot(r, h, n, &rho);
		DBGSCA("rho = <r, h> = ", rho);

		/* Normalize the residual with initial one */
		sc->residual= sqrt(rho) * bnrm2;


   	
		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		printf("res_%d=%e\n", iter+1, sc->residual);
		if(sc->residual <= sc->tolerance)
			break;


		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;
		DBGSCA("beta = rho / rho_old= ", beta);

		/* p(k+1)    = h(k+1) + beta*p(k) */
		xpay(h, beta, n, p);
		DBGVEC("p = h + beta * p> = ", p, n);

	}

	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = timeMatvec;

	/* Clean up */
	free(r);
	free(p);
	free(q);
	free(h);
}
