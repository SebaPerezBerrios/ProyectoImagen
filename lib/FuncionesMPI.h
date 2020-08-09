#ifndef FuncionesMPI_H
#define FuncionesMPI_H

#include <mpi.h>

#include <opencv2/core.hpp>
#include <vector>

using namespace cv;

void enviarImagenMPI(Mat& img, int destino) {
  // en caso de que no sea continua se genera una copia continua
  if (!img.isContinuous()) img = img.clone();
  int filas = img.rows;
  int columnas = img.cols;
  int dimension = img.elemSize();

  int dimensionDatos = filas * columnas * dimension;

  int dimensiones[3];
  dimensiones[0] = filas;
  dimensiones[1] = columnas;
  dimensiones[2] = dimension;

  MPI_Send(dimensiones, 3, MPI_INT, destino, 0, MPI_COMM_WORLD);
  MPI_Send(img.data, dimensionDatos, MPI_UNSIGNED_CHAR, destino, 0, MPI_COMM_WORLD);
}

Mat recibirImagenMPI(int origen, int tipoImagen = CV_8UC3) {
  MPI_Status status;

  int dimensiones[3];
  MPI_Recv(dimensiones, 3, MPI_INT, origen, 0, MPI_COMM_WORLD, &status);

  int filas = dimensiones[0];
  int columnas = dimensiones[1];
  int dimension = dimensiones[2];

  int dimensionDatos = filas * columnas * dimension;

  int* datosPtr = new int[dimensionDatos];

  MPI_Recv(datosPtr, dimensionDatos, MPI_UNSIGNED_CHAR, origen, 0, MPI_COMM_WORLD, &status);

  auto imagen = Mat(filas, columnas, tipoImagen, datosPtr).clone();
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

std::vector<int> particionar(int cantidad, int numeroParticiones) {
  int particion = cantidad / numeroParticiones;
  std::vector<int> particiones(numeroParticiones, particion);

  auto resto = cantidad - particion * numeroParticiones;
  for (auto& particion : particiones) {
    if (resto--)
      particion++;
    else
      break;
  }
  return particiones;
}

#endif