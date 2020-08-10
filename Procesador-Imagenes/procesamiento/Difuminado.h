#ifndef Difuminado_H
#define Difuminado_H

#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../lib/FuncionesMPI.h"

namespace difuminado {

// offset usado para el blur, se envian regiones anexas para tener un difuminado suave entre particiones
int Blur = 25;
int offset = Blur / 2;

void enviarImagen(int procesosReservados, int procesosTotales, const cv::Mat &imagenOriginal) {
  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);
  auto acumulado = vectorAcumulador(particiones);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    int offsetArriba = (proceso == 0) ? 0 : offset;
    int offsetAbajo = (proceso + 1 == procesosEsclavos) ? 0 : offset;

    auto regionAprocesar = cv::Rect(0, acumulado[proceso] - offsetArriba, imagenOriginal.cols,
                                    particiones[proceso] + offsetArriba + offsetAbajo);
    cv::Mat imagenAProcesar = imagenOriginal(regionAprocesar);

    enviarImagenMPI(imagenAProcesar, procesosReservados + proceso);
    enviarIntMPI(offsetArriba, procesosReservados + proceso);
    enviarIntMPI(offsetAbajo, procesosReservados + proceso);
  }
}

void procesarImagen() {
  auto imagenRecibida = recibirImagenMPI(0);
  int offsetArribaMPI = recibirIntMPI(0);
  int offsetAbajoMPI = recibirIntMPI(0);
  int hilos = omp_get_max_threads();
  cv::Mat nuevaImagen;

  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

  std::vector<cv::Mat> imagenesProcesadas(hilos);

#pragma omp for
  for (int proceso = 0; proceso < hilos; proceso++) {
    int offsetArriba = (proceso == 0) ? 0 : offset;
    int offsetAbajo = (proceso + 1 == hilos) ? 0 : offset;

    // generar sub imagen a ser procesada por el hilo
    auto regionAprocesar = cv::Rect(0, acumulado[proceso] - offsetArriba, imagenRecibida.cols,
                                    particiones[proceso] + offsetArriba + offsetAbajo);
    cv::Mat imagenAProcesar = imagenRecibida(regionAprocesar).clone();

    GaussianBlur(imagenAProcesar, imagenAProcesar, cv::Size(Blur, Blur), 0, 0);

    auto quitarOffset =
        cv::Rect(0, offsetArriba, imagenAProcesar.cols, imagenAProcesar.rows - offsetArriba - offsetAbajo);

    imagenesProcesadas[proceso] = imagenAProcesar(quitarOffset);
  }

  for (const auto &imagenProcesada : imagenesProcesadas) {
    nuevaImagen.push_back(imagenProcesada);
  }

  auto quitarOffset =
      cv::Rect(0, offsetArribaMPI, nuevaImagen.cols, nuevaImagen.rows - offsetArribaMPI - offsetAbajoMPI);

  auto imagenRecortada = nuevaImagen(quitarOffset);

  enviarImagenMPI(imagenRecortada, 0);
}
}  // namespace difuminado
#endif