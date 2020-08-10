#ifndef EscalaGrises_H
#define EscalaGrises_H

#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../lib/FuncionesMPI.h"

namespace escalaGrises {

using namespace cv;

void enviarImagen(int procesosReservados, int procesosTotales, const std::string& nombreArchivo) {
  std::string image_path = samples::findFile(nombreArchivo);
  Mat imagenOriginal = imread(image_path, IMREAD_UNCHANGED);
  cv::cvtColor(imagenOriginal, imagenOriginal, cv::COLOR_RGB2RGBA);

  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);
  auto acumulado = vectorAcumulador(particiones);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    auto regionAprocesar = Rect(0, acumulado[proceso], imagenOriginal.cols, particiones[proceso]);
    Mat imagenAProcesar = imagenOriginal(regionAprocesar);

    enviarImagenMPI(imagenAProcesar, procesosReservados + proceso);
  }
}

void procesarImagen() {
  auto imagenRecibida = recibirImagenMPI(0);
  int hilos = omp_get_max_threads();
  Mat nuevaImagen;

  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

#pragma omp for
  for (int proceso = 0; proceso < hilos; proceso++) {
    // generar sub imagen a ser procesada por el hilo
    auto regionAprocesar = Rect(0, acumulado[proceso], imagenRecibida.cols, particiones[proceso]);
    Mat imagenAProcesar = imagenRecibida(regionAprocesar);
    cvtColor(imagenAProcesar, imagenAProcesar, COLOR_BGR2GRAY);
#pragma omp critical
    nuevaImagen.push_back(imagenAProcesar);
  }

  enviarImagenMPI(nuevaImagen, 0);
}
}  // namespace escalaGrises
#endif