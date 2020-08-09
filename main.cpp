#include <mpi.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "lib/FuncionesMPI.h"

using namespace cv;
using namespace std;

#define Orquestador 0
#define Blur 21

void enviarImagen(int, int, const string&);
void recibirImagen();
void procesarImagen(int, int);
void unirImagen(int, int);
std::vector<int> particionar(int, int);

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
    enviarImagen(procesosReservados, procesosTotales, argv[1]);

    unirImagen(procesosReservados, procesosTotales);

    // participante();
  }

  if (mi_rango != Orquestador) { /* Esclavo */
    recibirImagen();
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

void enviarImagen(int procesosReservados, int procesosTotales, const std::string& nombreArchivo) {
  std::string image_path = samples::findFile(nombreArchivo);

  Mat imagenOriginal = imread(image_path, IMREAD_COLOR);

  // offset usado para el blur, se envian regiones anexas para tener un difuminado suave entre particiones
  int offset = Blur / 2;

  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    // calculo de offsets segun particion
    int offsetAbajo = (proceso + 1 == procesosEsclavos) ? 0 : offset;
    int offsetArriba = (proceso == 0) ? 0 : offset;
    int offsetResto = (proceso == 0) ? offset : 0;

    // generar sub imagen a ser enviada, se incluyen los offsets
    auto regionEnviada = Rect(0, 0, imagenOriginal.cols, particiones[proceso] + offsetAbajo + offsetArriba);
    Mat imagenEnviada = imagenOriginal(regionEnviada);

    // envio de imagenes a esclavos junto a los datos de particiones
    enviarImagenMPI(imagenEnviada, procesosReservados + proceso);
    enviarIntMPI(offsetAbajo, procesosReservados + proceso);
    enviarIntMPI(offsetArriba, procesosReservados + proceso);

    // quitar particion previamente enviada de imagen original
    auto regionRestante = Rect(0, particiones[proceso] - offsetResto, imagenOriginal.cols,
                               imagenOriginal.rows - particiones[proceso] + offsetResto);
    imagenOriginal = imagenOriginal(regionRestante);
  }
}

void recibirImagen() {
  auto imagenRecibida = recibirImagenMPI(0);
  int offsetAbajo = recibirIntMPI(0);
  int offsetArriba = recibirIntMPI(0);

  // aplicar difuminado, quitar offsets y retornar al orquestador
  GaussianBlur(imagenRecibida, imagenRecibida, Size(Blur, Blur), 0, 0);

  auto quitarOffset = Rect(0, offsetArriba, imagenRecibida.cols, imagenRecibida.rows - offsetAbajo - offsetArriba);

  auto imagenRecortada = imagenRecibida(quitarOffset);

  enviarImagenMPI(imagenRecortada, 0);
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