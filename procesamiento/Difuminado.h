#ifndef Difuminado_H
#define Difuminado_H

#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../lib/FuncionesMPI.h"

namespace difuminado {

#define Blur 25
int offset = Blur / 2;

void enviarImagen(int procesosReservados, int procesosTotales, const std::string& nombreArchivo) {
  std::string image_path = samples::findFile(nombreArchivo);
  Mat imagenOriginal = imread(image_path, IMREAD_UNCHANGED);
  cv::cvtColor(imagenOriginal, imagenOriginal, cv::COLOR_RGB2RGBA);

  // offset usado para el blur, se envian regiones anexas para tener un difuminado suave entre particiones

  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);
  auto acumulado = vectorAcumulador(particiones);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    int offsetArriba = (proceso == 0) ? 0 : offset;
    int offsetAbajo = (proceso + 1 == procesosEsclavos) ? 0 : offset;

    auto regionAprocesar = Rect(0, acumulado[proceso] - offsetArriba, imagenOriginal.cols,
                                particiones[proceso] + offsetArriba + offsetAbajo);
    Mat imagenAProcesar = imagenOriginal(regionAprocesar);

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
  Mat nuevaImagen;

  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

#pragma omp for
  for (int proceso = 0; proceso < hilos; proceso++) {
    int offsetArriba = (proceso == 0) ? 0 : offset;
    int offsetAbajo = (proceso + 1 == hilos) ? 0 : offset;

    // generar sub imagen a ser procesada por el hilo
    auto regionAprocesar = Rect(0, acumulado[proceso] - offsetArriba, imagenRecibida.cols,
                                particiones[proceso] + offsetArriba + offsetAbajo);
    Mat imagenAProcesar = imagenRecibida(regionAprocesar).clone();

    GaussianBlur(imagenAProcesar, imagenAProcesar, Size(Blur, Blur), 0, 0);

    auto quitarOffset = Rect(0, offsetArriba, imagenAProcesar.cols, imagenAProcesar.rows - offsetArriba - offsetAbajo);

    imagenAProcesar = imagenAProcesar(quitarOffset);

#pragma omp critical
    nuevaImagen.push_back(imagenAProcesar);
  }

  auto quitarOffset = Rect(0, offsetArribaMPI, nuevaImagen.cols, nuevaImagen.rows - offsetArribaMPI - offsetAbajoMPI);

  auto imagenRecortada = nuevaImagen(quitarOffset);

  enviarImagenMPI(imagenRecortada, 0);
}
}  // namespace difuminado
#endif