#ifndef FuncionesMPI_H
#define FuncionesMPI_H

#include <mpi.h>

#include <opencv2/core.hpp>
#include <vector>

void enviarImagenMPI(cv::Mat& img, int destino) {
  // en caso de que no sea continua se genera una copia continua
  if (!img.isContinuous()) img = img.clone();
  int filas = img.rows;
  int columnas = img.cols;
  int dimension = img.elemSize();
  int tipo = img.type();

  int dimensionDatos = filas * columnas * dimension;

  int dimensiones[4];
  dimensiones[0] = filas;
  dimensiones[1] = columnas;
  dimensiones[2] = dimension;
  dimensiones[3] = tipo;

  MPI_Send(dimensiones, 4, MPI_INT, destino, 0, MPI_COMM_WORLD);
  MPI_Send(img.data, dimensionDatos, MPI_UNSIGNED_CHAR, destino, 0, MPI_COMM_WORLD);
}

cv::Mat recibirImagenMPI(int origen) {
  MPI_Status status;

  int dimensiones[4];
  MPI_Recv(dimensiones, 4, MPI_INT, origen, 0, MPI_COMM_WORLD, &status);

  int filas = dimensiones[0];
  int columnas = dimensiones[1];
  int dimension = dimensiones[2];
  int tipo = dimensiones[3];

  int dimensionDatos = filas * columnas * dimension;

  char* datosPtr = new char[dimensionDatos];

  MPI_Recv(datosPtr, dimensionDatos, MPI_UNSIGNED_CHAR, origen, 0, MPI_COMM_WORLD, &status);

  auto imagen = cv::Mat(filas, columnas, tipo, datosPtr).clone();
  delete[] datosPtr;
  return imagen;
}

int recibirIntMPI(int origen) {
  MPI_Status status;
  int entero;
  MPI_Recv(&entero, 1, MPI_INT, origen, 0, MPI_COMM_WORLD, &status);
  return entero;
}

void enviarIntMPI(int entero, int destino) { MPI_Send(&entero, 1, MPI_INT, destino, 0, MPI_COMM_WORLD); }

#endif