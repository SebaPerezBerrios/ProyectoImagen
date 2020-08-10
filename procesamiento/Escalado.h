#ifndef Escalado_H
#define Escalado_H

#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../lib/FuncionesMPI.h"

namespace escalado {

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

void procesarImagen() {
  auto imagenRecibida = recibirImagenMPI(0);
  int hilos = omp_get_max_threads();
  cv::Mat nuevaImagen;

  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

#pragma omp for
  for (int proceso = 0; proceso < hilos; proceso++) {
    // generar sub imagen a ser procesada por el hilo
    auto regionAprocesar = cv::Rect(0, acumulado[proceso], imagenRecibida.cols, particiones[proceso]);
    cv::Mat imagenAProcesar = imagenRecibida(regionAprocesar);
    resize(imagenAProcesar, imagenAProcesar, cv::Size(), 1.33, 1.33);
#pragma omp critical
    nuevaImagen.push_back(imagenAProcesar);
  }

  enviarImagenMPI(nuevaImagen, 0);
}
}  // namespace escalado
#endif