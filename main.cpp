#include <mpi.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "lib/FuncionesMPI.h"
#include "procesamiento/Difuminado.h"
#include "procesamiento/EscalaGrises.h"
#include "procesamiento/Escalado.h"

using namespace cv;
using namespace std;

#define Orquestador 0

void unirImagen(int, int, const std::string &);
std::string obtenerTiempo();

int main(int argc, char **argv) {
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

  if (argc < 2) {
    std::cerr << "Se nececita una operacion y un nombre de imagen" << std::endl;
    return EXIT_FAILURE;
  }
  if (argc < 3) {
    std::cerr << "Se nececita un nombre de imagen" << std::endl;
    return EXIT_FAILURE;
  }

  auto tipoProceso = std::string(argv[1]);

  if (mi_rango == Orquestador) {
    if (tipoProceso == "1") {
      difuminado::enviarImagen(procesosReservados, procesosTotales, argv[2]);
    }
    if (tipoProceso == "2") {
      escalaGrises::enviarImagen(procesosReservados, procesosTotales, argv[2]);
    }

    if (tipoProceso == "3") {
      escalado::enviarImagen(procesosReservados, procesosTotales, argv[2]);
    }

    unirImagen(procesosReservados, procesosTotales, tipoProceso);
  }
  if (mi_rango != Orquestador) { /* Esclavo */
    if (tipoProceso == "1") difuminado::procesarImagen();
    if (tipoProceso == "2") escalaGrises::procesarImagen();
    if (tipoProceso == "3") escalado::procesarImagen();
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

void unirImagen(int procesosReservados, int procesosTotales, const std::string &tipoProceso) {
  auto imagenGenerada = Mat();
  int procesosEsclavos = procesosTotales - procesosReservados;

  auto tipoImagen = (tipoProceso == "2") ? CV_8UC1 : CV_8UC4;

  // crear nueva imagen a partir de particiones
  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    auto imagenRecibida = recibirImagenMPI(procesosReservados + proceso, tipoImagen);
    imagenGenerada.push_back(imagenRecibida);
  }

  std::stringstream nombreArchivo;
  nombreArchivo << setfill('0');
  nombreArchivo << "operacion_" << tipoProceso << "_" << obtenerTiempo() << ".png";

  imwrite(nombreArchivo.str(), imagenGenerada);
}

std::string obtenerTiempo() {
  auto tiempo = time(0);
  auto tiempoLocalPtr = std::localtime(&tiempo);
  stringstream tiempoLocal;
  tiempoLocal << setfill('0');
  tiempoLocal << std::setw(4) << tiempoLocalPtr->tm_year + 1900 << std::setw(2) << tiempoLocalPtr->tm_mon + 1
              << std::setw(2) << tiempoLocalPtr->tm_mday << std::setw(2) << tiempoLocalPtr->tm_hour << std::setw(2)
              << tiempoLocalPtr->tm_min << std::setw(2) << tiempoLocalPtr->tm_sec;
  return tiempoLocal.str();
}