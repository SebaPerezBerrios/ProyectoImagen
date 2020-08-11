#ifndef TransporteImagen_H
#define TransporteImagen_H

#include <omp.h>

#include "../lib/FuncionesMPI.h"
#include "../lib/Particionado.h"

void enviarImagen(int procesosReservados, int procesosTotales, const cv::Mat &imagenOriginal) {
  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);
  auto acumulado = vectorAcumulador(particiones);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    auto regionAprocesar = cv::Rect(0, acumulado[proceso], imagenOriginal.cols, particiones[proceso]);
    cv::Mat imagenAProcesar = imagenOriginal(regionAprocesar);

    enviarImagenMPI(imagenAProcesar, procesosReservados + proceso);
  }
}

void procesarImagen(const std::function<void(cv::Mat &)> &transformacion) {
  auto imagenRecibida = recibirImagenMPI(0);
  int hilos = omp_get_max_threads();
  cv::Mat nuevaImagen;

  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

  std::vector<cv::Mat> imagenesProcesadas(hilos);

#pragma omp for
  for (int proceso = 0; proceso < hilos; proceso++) {
    // generar sub imagen a ser procesada por el hilo
    auto regionAprocesar = cv::Rect(0, acumulado[proceso], imagenRecibida.cols, particiones[proceso]);
    auto imagenAProcesar = imagenRecibida(regionAprocesar);

    transformacion(imagenAProcesar);

    imagenesProcesadas[proceso] = imagenAProcesar;
  }

  for (const auto &imagenProcesada : imagenesProcesadas) {
    nuevaImagen.push_back(imagenProcesada);
  }

  enviarImagenMPI(nuevaImagen, 0);
}

#endif