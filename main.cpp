#include <mpi.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "lib/FuncionesMPI.h"
#include "procesamiento/Difuminado.h"
#include "procesamiento/Escalado.h"

using namespace cv;
using namespace std;

#define Orquestador 0

void unirImagen(int, int);

int main(int argc, char** argv) {
  int mi_rango;
  int procesosTotales;
  int procesosReservados = 1;
  int procesosMinimos = 2;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);

  MPI_Comm_size(MPI_COMM_WORLD, &procesosTotales);
  if (procesosTotales < procesosMinimos) {
    std::cerr << "ERROR, el programa necesita al menos 2 nodos de procesamiento" << std::endl;
    return EXIT_FAILURE;
  }

  if (mi_rango == Orquestador) {
    if (argc < 2) {
      std::cerr << "Se nececita un nombre de archivo de imagen" << std::endl;
      return EXIT_FAILURE;
    }

    // Enviar a nodos
    difuminado::enviarImagen(procesosReservados, procesosTotales, argv[1]);

    // Recibir y unir resultado
    unirImagen(procesosReservados, procesosTotales);

    // participante();
  }

  if (mi_rango != Orquestador) { /* Esclavo */
    difuminado::procesarImagen();
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

void unirImagen(int procesosReservados, int procesosTotales) {
  auto imagenGenerada = Mat();
  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear nueva imagen a partir de particiones
  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    auto imagenRecibida = recibirImagenMPI(procesosReservados + proceso);
    imagenGenerada.push_back(imagenRecibida);
  }

  imshow("Imagen procesada", imagenGenerada);
  waitKey(0);
}
