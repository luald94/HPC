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

#include "solver.h"
#include "output.h"

__device__ void unrollReduceCuda(volatile floatType *sdata, int tid){
	sdata[tid]+= sdata[tid + 32];
	sdata[tid]+= sdata[tid + 16];
	sdata[tid]+= sdata[tid + 8];
	sdata[tid]+= sdata[tid + 4];
	sdata[tid]+= sdata[tid + 2];
	sdata[tid]+= sdata[tid + 1];
}


__global__ void reduceCUDA(floatType* idata, floatType *odata, int n){
	 __shared__ floatType sdata[300];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	if(i<n){
		sdata[tid] = idata[i] + idata[i+blockDim.x];
	}else{
		sdata[tid] = 0;
	}
	
	__syncthreads();
	if(i < n){
		for(unsigned int s=blockDim.x/2; s>32; s>>=1){

			if(tid <s){
				sdata[tid] += sdata[tid+s];
			}
			__syncthreads();
		}
		if(tid < 32){
			unrollReduceCuda(sdata,tid);
		}
		if(tid == 0){
			odata[blockIdx.x] = sdata[0];
		}
		
	} 
}

/* ab <- a' * b */
void vectorDot(const floatType* a, const floatType* b, const int n, floatType* ab){
	int i;
	floatType temp;
	temp=0;
	for(i=0;i<n;i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

__global__ void vectorDotCUDA(const floatType* a, const floatType* b, const int n, floatType* reduction){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n){
		reduction[i]= a[i]*b[i];
	}
}

void reductionSum(const floatType* toReduce, const int n, floatType* sum){
	int i;
	floatType temp = 0;
	for(i = 0; i<n; i++)
		temp += toReduce[i];
	*sum = temp;
}

/* y <- ax + y */
__global__ void axpyCUDA(const floatType* a, const floatType* x, const int n, floatType* y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n){
		y[i]=a[0]*x[i]+y[i];
	}
}

/* y <- -ax + y */
__global__ void axpyNEG_CUDA(const floatType* a, const floatType* x, const int n, floatType* y){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i<n){
                y[i]=(-a[0])*x[i]+y[i];
        }
}

/* y <- x + ay */
__global__ void xpayCUDA(const floatType* x, const floatType* a, const int n, floatType* y){
	int i= blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n){
		y[i]=x[i]+a[0]*y[i];
	}
}

__global__ void xpyNegCUDA(const floatType* x, const int n, floatType* y){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		y[i] = x[i] - y[i];
	} 
}

/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
__global__ void matvecCUDA(const int n, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y){
	int i, j, k;
	floatType temp;	
	i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) {
		temp = 0;
		for (j = 0; j < length[i]; j++) {
			k = j * n + i;
			temp += data[k] * x[indices[k]];
		}
		y[i] = temp;
	}
}

/* y <- C*x where C is diag(A)^{-1} for Jacobi Preconditioner */
__global__ void matvec_diagCUDA(const int n, const floatType* data_diag, const floatType* x, floatType* y){
	int i;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		y[i] = data_diag[i] * x[i];
	}
}


/* nrm <- ||x||_2 */
void nrm2(const floatType* x, const int n, floatType* nrm){
	int i;
	floatType temp;
	temp = 0;
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
	floatType* r, *p, *q, *reduction, *h;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2;
	int iter;
	dim3 threadsPerBlock(128);
	dim3 blocksPerGrid(n/threadsPerBlock.x+1);
	dim3 blocksPerGridSecond(blocksPerGrid.x/threadsPerBlock.x+1);


	//double timeMatvec_s;
	double timeMatvec=0;
	float timeMatvec_i;

	/* declare CUDA_Events to time matvec */
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
		
	/* allocate memory */
	r = (floatType*)malloc(n * sizeof(floatType));
	p = (floatType*)malloc(n * sizeof(floatType));
	q = (floatType*)malloc(n * sizeof(floatType));
	reduction = (floatType*)malloc(n * sizeof(floatType));
	h = (floatType*)malloc(n * sizeof(floatType));

	/* declare GPU pointer */
	floatType *d_b,*d_reduction,*d_data,*d_r,*d_x,*d_p,*d_q,*d_alpha,*d_beta, *d_h, *d_data_diag;
	int *d_length,*d_indices;
	 
	/* allocate GPU memory */
	cudaMalloc(&d_data, (maxNNZ*n)*sizeof(floatType));
	cudaMalloc(&d_r, n*sizeof(floatType));
	cudaMalloc(&d_x, n*sizeof(floatType));
	cudaMalloc(&d_p, n*sizeof(floatType));
	cudaMalloc(&d_q, n*sizeof(floatType));
	cudaMalloc(&d_alpha, sizeof(floatType));
	cudaMalloc(&d_beta, sizeof(floatType));
	cudaMalloc(&d_length, n*sizeof(int));
	cudaMalloc(&d_indices, (maxNNZ*n)*sizeof(int));
	cudaMalloc(&d_b, n*sizeof(floatType));
	cudaMalloc(&d_reduction, n*sizeof(floatType));	
	cudaMalloc(&d_h, n*sizeof(floatType));
	cudaMalloc(&d_data_diag, n*sizeof(floatType));

	/* copy data to GPU */
	cudaMemcpy(d_data, data, (maxNNZ*n)*sizeof(floatType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, indices, (maxNNZ*n)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_length, length, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, n*sizeof(floatType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data_diag, data_diag, n*sizeof(floatType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n*sizeof(floatType), cudaMemcpyHostToDevice);

	DBGMAT("Start matrix A = ", n, nnz, maxNNZ, data, indices, length)
	DBGVEC("b = ", b, n);
	DBGVEC("x = ", x, n);


	cudaMemcpy(d_r, r, n*sizeof(floatType), cudaMemcpyHostToDevice);
	/* r(0)    = b - Ax(0) */
	cudaEventRecord(start);
	matvecCUDA<<<blocksPerGrid, threadsPerBlock>>>(n, d_data, d_indices, d_length, d_x, d_r);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeMatvec_i, start, stop);
	timeMatvec += timeMatvec_i;
		
	xpyNegCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_b, n, d_r);
	cudaThreadSynchronize();
	DBGVEC("r = b - Ax = ", r, n);
	
	cudaMemcpy(r, d_r, n*sizeof(floatType), cudaMemcpyDeviceToHost);
	
	/* h(0) = C * r(0) */
	matvec_diagCUDA<<<blocksPerGrid, threadsPerBlock>>>(n, d_data_diag, d_r, d_h);
	cudaMemcpy(h, d_h, n*sizeof(floatType), cudaMemcpyDeviceToHost);
	DBGVEC("h = C * r= ", h, n);

	nrm2(h, n, &bnrm2);
	bnrm2 = 1.0 /bnrm2;

	/* p(0)    = h(0) */
	memcpy(p, h, n*sizeof(floatType));
	DBGVEC("p = r = ", p, n);

	/* rho(0)    =  <r(0),r(0)> */
	vectorDot(r, h, n, &rho);
	printf("rho_0=%e\n", rho);

	/* copy more data to GPU */
	cudaMemcpy(d_p, p, n*sizeof(floatType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_q, q, n*sizeof(floatType), cudaMemcpyHostToDevice);
			
	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);

		/* q(k)      = A * p(k) */
		cudaEventRecord(start);
		matvecCUDA<<<blocksPerGrid, threadsPerBlock>>>(n, d_data, d_indices, d_length, d_p, d_q);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timeMatvec_i, start, stop);
		timeMatvec += timeMatvec_i;
		DBGVEC("q = A * p= ", q, n);
		
		/* dot_pq    = <p(k),q(k)> */
		vectorDotCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_p, d_q, n, d_reduction);
		cudaThreadSynchronize();
		reduceCUDA<<<blocksPerGrid.x/2, threadsPerBlock>>>(d_reduction, d_reduction,n);
		reduceCUDA<<<blocksPerGridSecond.x/2, threadsPerBlock>>>(d_reduction, d_reduction, blocksPerGrid.x);
		cudaMemcpy(reduction, d_reduction, blocksPerGridSecond.x*sizeof(floatType), cudaMemcpyDeviceToHost);
		reductionSum(reduction, blocksPerGridSecond.x, &dot_pq);

		DBGSCA("dot_pq = <p, q> = ", dot_pq);

		/* alpha     = rho(k) / dot_pq */
		alpha = rho / dot_pq;
		DBGSCA("alpha = rho / dot_pq = ", alpha);
		
		cudaMemcpy(d_alpha, &alpha, sizeof(floatType), cudaMemcpyHostToDevice);
		
		/* x(k+1)    = x(k) + alpha*p(k) */
		axpyCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_alpha, d_p, n, d_x);
		cudaThreadSynchronize();
		DBGVEC("x = x + alpha * p= ", x, n);

		/* r(k+1)    = r(k) - alpha*q(k) */
		axpyNEG_CUDA<<<blocksPerGrid, threadsPerBlock>>>(d_alpha, d_q, n, d_r);
		cudaThreadSynchronize();
		DBGVEC("r = r - alpha * q= ", r, n);

		/* h(k+1) = C * r(k+1) */
		matvec_diagCUDA<<<blocksPerGrid, threadsPerBlock>>>(n, d_data_diag, d_r, d_h);
		DBGVEC("h = C * r= ", h, n);

		rho_old = rho;
		DBGSCA("rho_old = rho = ", rho_old);

		/* rho(k+1)  = <r(k+1), h(k+1)> */
		vectorDotCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_r,d_h, n,d_reduction);
		cudaThreadSynchronize();
		reduceCUDA<<<blocksPerGrid.x/2, threadsPerBlock.x>>>(d_reduction, d_reduction,n);
		reduceCUDA<<<blocksPerGridSecond.x/2, threadsPerBlock.x>>>(d_reduction, d_reduction, blocksPerGrid.x);
		cudaMemcpy(reduction, d_reduction, blocksPerGridSecond.x*sizeof(floatType), cudaMemcpyDeviceToHost);
		reductionSum(reduction, blocksPerGridSecond.x, &rho);
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
		cudaMemcpy(d_beta, &beta, sizeof(floatType), cudaMemcpyHostToDevice);

		/* p(k+1)    = h(k+1) + beta*p(k) */
		xpayCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_h, d_beta, n, d_p);
	       	cudaThreadSynchronize();
		DBGVEC("p = h + beta * p> = ", p, n);
	}

	/* copy back result */
	cudaMemcpy(x, d_x, n*sizeof(floatType), cudaMemcpyDeviceToHost);

	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = timeMatvec/1000;

	/* Clean up */
	free(r);
	free(p);
	free(q);
	free(h);
	cudaFree(&d_x);
	cudaFree(&d_r);
	cudaFree(&d_p);
	cudaFree(&d_data);
	cudaFree(&d_alpha);
	cudaFree(&d_beta);
	cudaFree(&d_length);
	cudaFree(&d_indices);
	cudaFree(&d_b);
	cudaFree(&d_h);
}
