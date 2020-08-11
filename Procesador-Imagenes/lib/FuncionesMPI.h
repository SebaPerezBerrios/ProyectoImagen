#ifndef FuncionesMPI_H
#define FuncionesMPI_H

#include <mpi.h>

#include <opencv2/core.hpp>

/**
 * Funcion que envia una representacion de imagen de OpenCV como un string de bytes junto a la informacion necesaria
 * para reconstruir dicha imagen
 *
 * @param imagen imagen a ser enviada.
 * @param destino nodo donde sera enviada la imagen.
 */
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

/**
 * Funcion que recibe unn string de bytes junto a la informacion necesaria
 * para reconstruir la imagen original.
 *
 * @param origen nodo donde proviene la imagen.
 * @returns imagen generada.
 */
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

/**
 * Funcion que recibe un entero de un nodo
 *
 * @param origen nodo donde proviene el entero.
 * @returns entero recibido.
 */
int recibirIntMPI(int origen) {
  MPI_Status status;
  int entero;
  MPI_Recv(&entero, 1, MPI_INT, origen, 0, MPI_COMM_WORLD, &status);
  return entero;
}

/**
 * Funcion que envia un entero a un nodo.
 *
 * @param entero entero a ser enviada.
 * @param destino nodo donde sera enviado el entero.
 */
void enviarIntMPI(int entero, int destino) { MPI_Send(&entero, 1, MPI_INT, destino, 0, MPI_COMM_WORLD); }

#endif