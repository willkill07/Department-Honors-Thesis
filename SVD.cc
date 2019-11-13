/*
  Author     : Maksim Y. Melnichenko
  Title      : SVD Inplementation

  Project history and additional details are avaliable upon request.
/************************************************************/
// System includes

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <utility>
#include <vector>

//custom matrix header
#include "Matrix.h"

/************************************************************/
// Using declarations

using std::cin;
using std::cout;
using std::endl;
using std::pow;
using std::vector;
using std::sqrt;
using std::pow;

using Correction = std::pair<double, Matrix>;
using MatrixPair = std::pair<Matrix, Matrix>;

/************************************************************/
// Function prototypes/global vars/typedefs

void build(Matrix &A);

void tridiagonalizer(Matrix &B);

MatrixPair divideNConquer(Matrix &B);

Correction block_diagonal(Matrix &B);

Matrix secular_solver(const Matrix &diag, const Correction &Beta);

Matrix initial_e_approx(const Matrix &diag, const Correction &Beta);


//merge together
double g_secular(double x, int k, const Matrix&  D, const Correction & Beta);

double f_secular(double x, const Matrix&  D, const Correction & Beta);

double f_prime_secular(double x, const Matrix&  D, const Correction & Beta);

double split_secular_1_prime(double x, int k, const Matrix&  D, const Correction & Beta);

double split_secular_2_prime(double x, int k, const Matrix&  D, const Correction & Beta);

template <bool O, bool T>
void printMatrix(const MatrixT<O,T> &A);

/************************************************************/
// Main function

int main(int argc, char *argv[]) {

  // int r = 2;
  // int c = 4;

  // if ( r < c)
  //{
  //    int i = r;
  //    r = c;
  //    c = i;
  //}
  // Since (A^T) * A == A * (A^T)

  // Matrix A (r, c);

  // build(A);
  // Initial Matrix
  // printMatrix(A);

  // auto B = A * A.transpose();

  // Building a numerical example

  Matrix TEST(4, 4);

  /*
  TEST(0, 0) = 4;
  TEST(0, 1) = 1;
  TEST(0, 2) = -2;
  TEST(0, 3) = 2;

  TEST(1, 0) = 1;
  TEST(1, 1) = 2;
  TEST(1, 2) = 0;
  TEST(1, 3) = 1;

  TEST(2, 0) = -2;
  TEST(2, 1) = 0;
  TEST(2, 2) = 3;
  TEST(2, 3) = -2;

  TEST(3, 0) = 2;
  TEST(3, 1) = 1;
  TEST(3, 2) = -2;
  TEST(3, 3) = -1;
  */
  // A * (A ^ T)

  TEST(0, 0) = 7;
  TEST(0, 1) = 3;
  TEST(0, 2) = 0;
  TEST(0, 3) = 0;

  TEST(1, 0) = 3;
  TEST(1, 1) = 1;
  TEST(1, 2) = 2;
  TEST(1, 3) = 0;

  TEST(2, 0) = 0;
  TEST(2, 1) = 2;
  TEST(2, 2) = 8;
  TEST(2, 3) = -2;

  TEST(3, 0) = 0;
  TEST(3, 1) = 0;
  TEST(3, 2) = -2;
  TEST(3, 3) = 3;
  
  //tridiagonalizer(TEST);

  /*TEST DNC - 1
  Matrix l (2, 2);
  l(0, 0) = 1;
  l(0, 1) = 3;
  l(1, 0) = 3;
  l(1, 1) = 2;
  
  */

  
  printMatrix(TEST);
  auto [ortho, eigenvalues] = divideNConquer(TEST);
  printMatrix(eigenvalues);
  /*
  //TEST 2 - Secular Solver
  
  Matrix l (4, 4);
  l(0, 0) = 1;
  l(1, 1) = 3;
  l(2, 2) = 5;
  l(3, 3) = 7;

  Matrix c (4, 1);
  c(0, 0) = 0.5;
  c(1, 0) = 0.7;
  c(2, 0) = 0.4;
  c(3, 0) = 0.2;

  Correction Beta = std::make_pair(6, c);

  secular_solver(l, Beta);
  
/*
// TEST 3 - Secular Solver

 Matrix l (2, 2);
  l(0, 0) = 1;
  l(1, 1) = 2;
  

  Matrix c (2, 1);
  c(0, 0) = 3;
  c(1, 0) = 4;

  Correction Beta = std::make_pair(6, c);

  secular_solver(l, Beta);
  */


}


/*
Build - by reference, modifies the original matrix. Assignes a random
floating point number to each of its cells.
*/
void build(Matrix &A) {
  
  static std::minstd_rand rng{9999};
  static std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (int i = 0; i < A.rows(); ++i) 
  {
    for (int j = 0; j < A.colms(); ++j) 
    {
      A(i, j) = dist (rng);
    }
  }
}


/*
Tridiagonalizer - Implementation of Householder's method of turning a
symmetric matrix into a symmetric tridiagonal one. Modifies the original by refernce,
uses a helper routine for the similarity transformation matrix production.
*/


void tridiagonalizer(Matrix &B) {
  // producing a column vector, replicating the last column of B

  double alpha, RSQ;
  const int n = B.rows();
  
  for (int k = 0; k < n - 2; ++k) {

    alpha = RSQ = 0;

    Matrix W(B.rows(), 1);
    // for k = 0 ...  < n-2

    for (int i = k + 1; i < n; ++i) 
    {
      alpha += std::pow(B(i, k), 2);
      W(i, 0) = (0.5) * B(i, k);
    }

    const double leadingVal = B(k + 1, k);
    //final alpha definition
    alpha = -(std::sqrt(alpha)); 
    //represents 2r^2
    RSQ = std::pow(alpha, 2) - (alpha * leadingVal); 
    
    //leading entry in w-vector
    W(k + 1, 0) = (0.5) * (leadingVal - alpha); 

    //producting a similarity transformation
    auto WTW = W * W.transpose(); 
    WTW = Matrix::identity(n) + ((-4 / RSQ) * WTW);
    //transforming the original matrix
    B = WTW * B * WTW;

  }
}





/*
Divide - initial step of Cuppen's Divide and Conquer Eigenvalue Extraction algorithm.
/////////////////////////////////UNDER CONSTRUCTION/////////////////////////////////
*/
MatrixPair divideNConquer(Matrix &B)
{
  int n = B.rows();

  if (n == 2) 
  {

    double a  = B(0, 0);
    double d = B(1, 1);
    double c  = B(1, 0);
    double l1, l2;

    Matrix ortho (n, n);
    Matrix diag  (n, n);


    l1 = diag(0, 0) = ((a + d) + sqrt( pow((a + d), 2) - (4 * ((a * d) - pow(c, 2))))) / 2;
    l2 = diag(1, 1) = ((a + d) - sqrt( pow((a + d), 2) - (4 * ((a * d) - pow(c, 2))))) / 2;
    
    //eigenvector magnitudes
    double v12 = ((l1 - a) / c);
    double v22 = ((l2 - a) / c);
    double v1m = sqrt( 1 + pow( v12, 2));
    double v2m = sqrt( 1 + pow( v22, 2));

    ortho(0, 0) =   1 / v1m;
    ortho(0, 1) =   1 / v2m;
    ortho(1, 0) = v12 / v1m;
    ortho(1, 1) = v22 / v2m;
    
    return MatrixPair(ortho, diag);
  } 
  else 
  {
    Correction Beta = block_diagonal(B);

    Matrix hi = B.cut( n / 2, 1);
    Matrix lo = B.cut(n - (n / 2), 0);
    
    const MatrixPair & hiNode = divideNConquer(hi);
    const MatrixPair & loNode = divideNConquer(lo);
    

    const auto & [o1, d1] = hiNode;
    const auto & [o2, d2] = loNode;

    Matrix ortho  = Matrix::combine (o1, o2);
    auto orthoT = ortho.transpose();
    Matrix diag   = Matrix::combine (d1, d2);

    const auto & [scalar, unitVector] = Beta;

    Matrix C = (1 / (sqrt(2))) * (orthoT * unitVector);
    Beta = std::make_pair(2 * scalar, C);

    auto corr = Beta.first * ( Beta.second * Beta.second.transpose());
    auto sec = diag + corr;
    auto thir = ortho * sec * orthoT;

    cout <<"This has to be equal the original\n";
    printMatrix(thir);

    return MatrixPair (ortho, secular_solver(diag, Beta));
  }
}


/*************************************
 *************************************
 *************************************
 *************************************
 *************************************
 *************************************
 *************************************/


Matrix secular_solver( const Matrix & diag, const Correction & Beta)
{
  const double n = diag.rows();
  const double epsilon = pow(10, -8);
  const auto & [P, Z] = Beta;
  Matrix l = initial_e_approx(diag, Beta);

  for(int i = 0; i < n; ++i)
  {
    if (i == n - 1)
    {
      //The case k = n is exactly the same as in 2.2.2
    }
    else 
    {
      double y = l(i, i);
      double delta = diag(i, i) - y;
      double delta1 = diag(i + 1, i + 1) - y;
      double f = f_secular(y, diag, Beta);
      double f_prime = f_prime_secular(y, diag, Beta);
      double fs1_prime = split_secular_1_prime(y, i, diag, Beta);
      double fs2_prime = split_secular_2_prime(y, i, diag, Beta);
      double t = 0;
      
      //I would assume that it is barbaric to split the computation of a and b
       
      double a = ((delta + delta1) * f) - (delta * delta1 * f_prime);
      double b = delta * delta1 * f;
      double c = f - (delta * fs1_prime) - (delta1 * fs2_prime);

      if (a > 0)
      {
        t = (a - sqrt((a * a) - (4 * b * c))) / (2 * c); 
      }
      else
      {
        t = (2 * b) / (a + sqrt((a * a) - (4 * b * c)));
      }
      
      l(i, i) = t + diag(i, i);
    }
  }
  
}


Matrix initial_e_approx(const Matrix & diag, const Correction & Beta)
{
  int n = diag.rows();
  const auto & [p, Z] = Beta;
  Matrix D = Matrix::diagSort(diag);
  Matrix l (n, n);

  for(int i = 0; i < n; ++i)
  {
    if (i == n - 1)
    {
      //Issue with n + 1 in the algorithm description
    }
    else 
    {
      double midpoint = (D(i + 1, i + 1) + D(i, i)) / 2;
      double delta = D(i + 1, i + 1) - D(i, i);
      double c = g_secular(midpoint, i, diag, Beta);
      double f = f_secular(midpoint, diag, Beta);
      const double z1 = Z(i, 0);
      const double z2 = Z(i + 1, 0);
      double a, b, t, k;
      
      //I would assume that it is barbaric to split the computation of a and b
      if ( f >= 0)
      {
        // t = y - D(k, k)
        k = i;
        a = (c * delta) + ((z1 * z1) + (z2 * z2));
        b = (z1 * z1) * delta;
      } 
      else
      {
        // t = y - D(k + 1, k + 1)
        k = i + 1;
        a = -(c * delta) + ((z1 * z1) + (z2 * z2));
        b = -(z1 * z1) * delta;
      }

      if (a > 0)
      {
        t = (a - sqrt((a * a) - (4 * b * c))) / (2 * c); 
      }
      else
      {
        t = (2 * b) / (a + sqrt((a * a) - (4 * b * c)));
      }
      
      l(i, i) = t + D(k, k);
    }
  }


  return l;
}

double g_secular(double x, int k, const Matrix&  D, const Correction & Beta)
{
  /*
  Difference in interpretation: artcle does somthing like division of secular equation by p,
  therefore, they are callig p what in my case equals 1/p
  */
  const double n = D.rows();
  const auto & [p, Z] = Beta;

  double res = 0.0;
  for(int j = 0; j < n; ++j)
  {
    if(j != k || j != k + 1)
    {
      const double z = Z(j, 0);
      res += (z * z / (D(j, j) - x));
    }  
  }
  return (1 / p) + res;
}

double f_secular(double x, const Matrix&  D, const Correction & Beta)
{
  const double n = D.rows();
  const auto & [p, Z] = Beta;

  double res = 0.0;
  for(int j = 0; j < n; ++j)
  {
      const double z = Z(j, 0);
      res += (z * z / (D(j, j) - x));
  }
  return (1 / p) + res;
}


double f_prime_secular(double x, const Matrix&  D, const Correction & Beta)
{
  const double n = D.rows();
  const auto & [p, Z] = Beta;

  double res = 0.0;
  for(int j = 0; j < n; ++j)
  {
      const double z = Z(j, 0);
      const double r = (D(j, j) - x);
      res += -(z * z / r * r);
  }
  return (1 / p) + res;
}

//0 ... k
double split_secular_1_prime(double x, int k, const Matrix&  D, const Correction & Beta)
{
  const auto & [p, Z] = Beta;

  double res = 0.0;
  for(int j = 0; j < k; ++j)
  {
      const double z = Z(j, 0);
      const double r = (D(j, j) - x);
      res += -(z * z / r * r);
  }
  return (1 / p) + res;
}

//k + 1 ... n
double split_secular_2_prime(double x, int k, const Matrix&  D, const Correction & Beta)
{
  const double n = D.rows();
  const auto & [p, Z] = Beta;

  double res = 0.0;
  for(int j = k + 1; j < n; ++j)
  {
      const double z = Z(j, 0);
      const double r = (D(j, j) - x);
      res += -(z * z / r * r);
  }
  return (1 / p) + res;
}

/*************************************
 *************************************
 *************************************
 *************************************
 *************************************
 *************************************
 *************************************/

/*
block_diagonal - routine that makes the original matrix, taken in by reference, 
block diagonal and additionally updates the "factored-out" matrix beta with corresponding
elements.
*/

Correction
block_diagonal(Matrix &B)
{
  int n = B.rows();
  Matrix Beta (n, 1);

  double m = n / 2;
  double beta_value = B(m, m - 1);
  
  Beta(m , 0) = Beta(m - 1, 0) = 1;

  B(m, m - 1) = B(m - 1, m) = 0;
  B(m, m) -= beta_value;
  B(m - 1, m - 1) -= beta_value;

  return std::make_pair(beta_value, Beta);
}


/*
Printing - Simple routine, created for the testing purposes.
*/
template <bool O, bool T>
void printMatrix(const MatrixT<O,T> &A) 
{
  for (int i = 0; i < A.rows(); ++i) 
  {
    for (int j = 0; j < A.colms(); ++j) 
    {
      printf("%13.5f", A(i, j));
    }
    puts("");
  }
  puts("");
}
