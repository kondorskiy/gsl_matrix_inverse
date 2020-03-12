
/*====================================================================

  Adoptation of the GSL matrix inverse code.

  Alexey D. Kondorskiy,
  P.N.Lebedev Physical Institute of the Russian Academy of Science.
  E-mail: kondorskiy@lebedev.ru, kondorskiy@gmail.com.

====================================================================*/

// For tests
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <complex>
#include <string>
#include <math.h>
#include <sys/stat.h>
#include <vector>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>


/*--------------------------------------------------------------------
  Matrix inverse using GSL.
--------------------------------------------------------------------*/
void matrixInverse(
  const int &n,  // Matrix dimension.
  double **a)    // Input and output matrix.
{
  gsl_matrix *mat = gsl_matrix_alloc(n, n);
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
      gsl_matrix_set(mat, i, j, a[i][j]);

  int s;
  gsl_permutation *lu = gsl_permutation_alloc(n);
  gsl_linalg_LU_decomp(mat, lu, &s);
  gsl_matrix *imat = gsl_matrix_alloc(n, n);
  gsl_linalg_LU_invert(mat, lu, imat);

  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
      a[i][j] = gsl_matrix_get(imat, i, j);

  gsl_matrix_free(mat);
  gsl_permutation_free(lu);
  gsl_matrix_free(imat);
}


/*--------------------------------------------------------------------
  3x3 matrix inverse.
--------------------------------------------------------------------*/
void matrixInverse3D(double (&a)[3][3])
{
  double **m = new double*[3];
  for(int i = 0; i < 3; ++i)
    m[i] = new double[3];

  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
      m[i][j] = a[i][j];

  matrixInverse(3, m);

  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
      a[i][j] = m[i][j];

  for(int i = 0; i < 3; ++i)
    delete [] m[i];
  delete [] m;
}


/*********************************************************************
  matrixInverse3D test
*********************************************************************/
int main(int argc, char **argv)
{
  double nn[3][3] = {
    { 0.20, -21.0, 0.011},
    { 0.1, -0.27, 10.0 },
    { 10.5, 0.32, 0.87} };

  double nn_inv[3][3], res;

  std::cout << "Initial matrix\n";
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      nn_inv[i][j] = nn[i][j];
      std::cout << nn[i][j] << " ";
    }
    std::cout << "\n";
  }
  matrixInverse3D(nn_inv);

  std::cout << "\nTest result should be unity\n";
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      res = 0.0;
      for(int k = 0; k < 3; k++)
        res = res + nn[i][k]*nn_inv[k][j];
      std::cout << res << " ";
    }
    std::cout << "\n";
  }

  return 0;
}   // */


/*********************************************************************
  matrixInverse test
**********************************************************************
int main(int argc, char **argv)
{
  double nn[5][5] = {
    { 0.20, -21.0, 0.011,  119, -1.01 },
    {  0.1, -0.27,  10.0, 0.78,  56.9 },
    {  3.1,  10.2, -11.0,  0.8,   6.9 },
    { -6.7,   0.7,  10.2, -7.8,  -5.6 },
    { 10.5,  0.32,  0.87, -3.7,   6.9 } };

  double **m = new double*[5];
  for(int i = 0; i < 5; ++i)
    m[i] = new double[5];

  std::cout << "\nInitial matrix\n";
  for(int i = 0; i < 5; i++) {
    for(int j = 0; j < 5; j++) {
      m[i][j] = nn[i][j];
      std::cout << nn[i][j] << " ";
    }
    std::cout << "\n";
  }
  matrixInverse(5, m);

  std::cout << "\nTest result should be unity\n";
  double nd_sum = 0.0;
  for(int i = 0; i < 5; i++) {
    for(int j = 0; j < 5; j++) {
      double res = 0.0;
      for(int k = 0; k < 5; k++)
        res = res + nn[i][k]*m[k][j];
      std::cout << res << " ";
      if(i != j) nd_sum = nd_sum + res;
    }
    std::cout << "\n";
  }

  std::cout << "\nDiagonal elements:\n";
  for(int i = 0; i < 5; i++) {
    double res = 0.0;
    for(int k = 0; k < 5; k++)
      res = res + nn[i][k]*m[k][i];
    std::cout << res << " ";
  }
  std::cout << "\nSum of nondiagonal elements: " << nd_sum << "\n\n";

  for(int i = 0; i < 5; ++i)
    delete [] m[i];
  delete [] m;

  return 0;
}   // */


//====================================================================
