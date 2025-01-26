#include "Hermite.h"

/**
 * @brief Computes the Hermite-Gauss quadrature nodes using the Golub-Welsch algorithm.
 * 
 * This function computes the nodes for Gauss-Hermite quadrature, which are the roots
 * of the Hermite polynomial. The algorithm constructs a symmetric tridiagonal Jacobi
 * matrix, computes its eigenvalues using the LAPACK routine `dstev_`, and sorts the
 * resulting nodes in ascending order.
 * 
 * @param N The number of quadrature nodes to compute (order of the Hermite polynomial).
 * @return A pointer to a dynamically allocated array of size N containing the computed
 *         quadrature nodes sorted in ascending order. Returns nullptr if memory allocation
 *         or eigenvalue computation fails.
*/

double* Nodes(int N){

    // x diagonal, y subdiagonal
    double* x = (double *)malloc(N*sizeof(double));
    double* y = (double *)malloc((N-1)*sizeof(double));

    if (!x || !y){
        free(x);
        free(y);
        std::cerr << "Error: Hermite nodes memory allocation failed." << std::endl;
        return nullptr;
    }

    for (int n=0; n<N-1; n++){
        x[n] = 0.0;
        y[n] = std::sqrt((double)(n+1)/2.0);
    }
    x[N-1] = 0.0;

    // LAPACK parameters
    char JOBZ = 'N';
    int  LDZ  = 1;
    double* dummy_ptr = nullptr;
    double WORK[1];
    int INFO;

    // Compute Eigenvalues
    dstev_(&JOBZ,&N,x,y,dummy_ptr,&LDZ,WORK,&INFO);

    free(y);
    if (INFO == 0 ){
        std::sort(x, x + N);
        return x;
    } else {
        free(x);
        std::cerr << "Error: Hermite nodes DSTEV failed with INFO = " << INFO << std::endl;
        return nullptr;
    }

}

/**
 * @brief Computes the Vandermonde matrix.
 * 
 * This function compute the vandermonde matrix based on an set of points x.
 * The vector x has dimension N and the matrix V has dimension NxM.
 * 
 * @param x nodes.
 * @param N number of elements in x.
 * @param M number of basis functions / columns in V.
 * 
 * @return A pointer to a dynamically allocated array of size N*M.
 *         this is the matrix V.
*/


double* Vander(double* x, int N, int M){

    // allocate memory for NxM matrix V
    double* V = (double *)malloc(N*M*sizeof(double));

    if (!V){
        free(V);
        std::cerr << "Error: Vander memory allocation failed." << std::endl;
        return nullptr;
    }

    // Constants
    double inv_pi_quarter = std::pow(std::numbers::pi, -0.25);
    double sqrt2 = std::numbers::sqrt2;

    // Compute first column of V
    for (int i=0; i<N; i++){
        V[i] = inv_pi_quarter * std::exp(-0.5*x[i]*x[i]);
    }

    if (M == 1){
        return V;
    }

    // Compute second column of V
    for (int i=0; i<N; i++){
        V[i+N] = sqrt2*x[i]*V[i];
    }

    if (M == 2){
        return V;
    }

    // Compute remaining columns of V using recursion relations
    for (int j=1; j<M-1; j++){
        double c1 = std::sqrt(j/2.0);
        double c2 = std::sqrt(2.0/(j+1.0));
        for (int i=0; i<N; i++){
            V[i+N*(j+1)] = c2*(x[i]*V[i+N*j]-c1*V[i+N*(j-1)]);
        }
    }

    return V;

}

/**
 * @brief Computes the derivative of Vandermonde matrix.
 * 
 * This function compute the derivative of vandermonde matrix, Vx, based
 * on an set of points x and a vandermonde matrix V also based on x.
 * The vector x has dimension N and the matrix V has dimensions NxM.
 * the derivative of vandermonde matrix Vx has dimensions NxM.
 * 
 * @param x nodes.
 * @param N number of elements in x.
 * @param M number of basis functions / columns in V.
 * 
 * @return A pointer to a dynamically allocated array of size N*M.
 *         this is the matrix Vx.
*/

double* VanderDiff(double* x, double* V, int N, int M){

    // allocate memory for NxM matrix V
    double* Vx = (double *)malloc(N*M*sizeof(double));

    if (!Vx){
        free(Vx);
        std::cerr << "Error: VanderDiff memory allocation failed." << std::endl;
        return nullptr;
    }

    // Constant
    double sqrt2 = std::numbers::sqrt2;

    // Compute first column of V
    for (int i=0; i<N; i++){
        Vx[i] = -x[i] * V[i];
    }

    if (M == 1){
        return Vx;
    }

    // Compute second column of V
    for (int i=0; i<N; i++){
        Vx[i+N] = (2.0/sqrt2)*V[i]-x[i]*V[i+N];
    }

    if (M == 2){
        return Vx;
    }

    // Compute remaining columns of V using recursion relations
    for (int j=1; j<M-1; j++){
        double c = 2.0*std::sqrt((j+1)/2.0);
        for (int i=0; i<N; i++){
            Vx[i+N*(j+1)] = c*V[i+N*j]-x[i]*V[i+N*(j+1)];
        }
    }

    return Vx;

}