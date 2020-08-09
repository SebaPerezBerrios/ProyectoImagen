#ifndef Escalado_H
#define Escalado_H

#include <omp.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "../lib/FuncionesMPI.h"

namespace escalado {

using namespace cv;
using namespace std;

void enviarImagen(int procesosReservados, int procesosTotales, const std::string& nombreArchivo) {
  std::string image_path = samples::findFile(nombreArchivo);

  Mat imagenOriginal = imread(image_path, IMREAD_UNCHANGED);
  cv::cvtColor(imagenOriginal, imagenOriginal, cv::COLOR_RGB2RGBA);

  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    // generar sub imagen a ser enviada
    auto regionEnviada = Rect(0, 0, imagenOriginal.cols, particiones[proceso]);
    Mat imagenEnviada = imagenOriginal(regionEnviada);

    // envio de imagenes a esclavos
    enviarImagenMPI(imagenEnviada, procesosReservados + proceso);

    // quitar particion previamente enviada de imagen original
    auto regionRestante =
        Rect(0, particiones[proceso], imagenOriginal.cols, imagenOriginal.rows - particiones[proceso]);
    imagenOriginal = imagenOriginal(regionRestante);
  }
}

void procesarImagen() {
  auto imagenRecibida = recibirImagenMPI(0);

  int hilos = omp_get_max_threads();
  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

  Mat nuevaImagen;

#pragma omp for
  for (int proceso = 0; proceso < hilos; proceso++) {
    // generar sub imagen a ser procesada por el hilo
    auto regionAprocesar = Rect(0, acumulado[proceso], imagenRecibida.cols, particiones[proceso]);
    Mat imagenAProcesar = imagenRecibida(regionAprocesar);
    resize(imagenAProcesar, imagenAProcesar, cv::Size(), 1.33, 1.33);
#pragma omp critical
    nuevaImagen.push_back(imagenAProcesar);
  }

  enviarImagenMPI(nuevaImagen, 0);
}
}  // namespace escalado
#endif