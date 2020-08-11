#ifndef TransporteImagen_H
#define TransporteImagen_H

#include <omp.h>

#include "../lib/FuncionesMPI.h"
#include "../lib/Particionado.h"

/**
 * Funcion que particiona y envia una imagen a travez de MPI a los nodos esclavos
 *
 * @param procesosReservados cantidad de nodos reservados.
 * @param procesosTotales cantidad de nodos totales.
 * @param imagenOriginal imagen que sera subdividida y enviada a los nodos esclavos.
 */
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

/**
 * Funcion que recibe una particion de imagen y aplica una transfrmacion sobre esta, luego retorna el resultado al nodo
 * orquestador
 *
 * @param transformacion funcion que aplica una transformacion a una imagen, la funcion no debe depender de pixeles
 * anexos.
 */
void procesarImagen(const std::function<void(cv::Mat &)> &transformacion) {
  auto imagenRecibida = recibirImagenMPI(0);
  int hilos = omp_get_max_threads();
  cv::Mat nuevaImagen;

  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

  std::vector<cv::Mat> imagenesProcesadas(hilos);

#pragma omp parallel for
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