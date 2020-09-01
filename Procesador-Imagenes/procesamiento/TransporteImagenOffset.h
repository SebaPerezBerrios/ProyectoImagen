#ifndef TransporteImagenOffset_H
#define TransporteImagenOffset_H

#include <omp.h>

#include "../lib/FuncionesMPI.h"
#include "../lib/Particionado.h"

/**
 * Funcion que particiona y envia una imagen a travez de MPI a los nodos esclavos
 *
 * @param procesosReservados cantidad de nodos reservados.
 * @param procesosTotales cantidad de nodos totales.
 * @param imagenOriginal imagen que sera subdividida y enviada a los nodos esclavos junto a un offset usado en la
 * transformacion posterior.
 */
void enviarImagenOffset(int procesosReservados, int procesosTotales, int offset, const cv::Mat &imagenOriginal) {
  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por columna
  auto particiones = particionar(imagenOriginal.cols, procesosEsclavos);
  auto acumulado = vectorAcumulador(particiones);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    int offsetIzq = (proceso == 0) ? 0 : offset;
    int offsetDer = (proceso + 1 == procesosEsclavos) ? 0 : offset;

    auto regionAprocesar =
        cv::Rect(acumulado[proceso] - offsetIzq, 0, particiones[proceso] + offsetIzq + offsetDer, imagenOriginal.rows);
    cv::Mat imagenAProcesar = imagenOriginal(regionAprocesar);

    enviarImagenMPI(imagenAProcesar, procesosReservados + proceso);
    enviarIntMPI(offsetIzq, procesosReservados + proceso);
    enviarIntMPI(offsetDer, procesosReservados + proceso);
    enviarIntMPI(offset, procesosReservados + proceso);
  }
}

/**
 * Funcion que recibe una particion de imagen junto a pixeles extra y aplica una transfrmacion sobre esta, luego retorna
 * el resultado al nodo orquestador sin los pixeles extra.
 *
 * @param transformacion funcion que aplica una transformacion a una imagen, la funcion puede hacer uso de los pixeles
 * extra para su computo, ejemplo la funcion de difuminado.
 */
void procesarImagenOffset(const std::function<void(cv::Mat &, int intensidad)> &transformacion) {
  auto imagenRecibida = recibirImagenMPI(0);
  int offsetIzqMPI = recibirIntMPI(0);
  int offsetDerMPI = recibirIntMPI(0);

  int offset = recibirIntMPI(0);
  int intensidad = offset * 2 + 1;

  int hilos = omp_get_max_threads();
  cv::Mat nuevaImagen;

  auto particiones = particionar(imagenRecibida.rows, hilos);
  auto acumulado = vectorAcumulador(particiones);

  std::vector<cv::Mat> imagenesProcesadas(hilos);

#pragma omp parallel for
  for (int proceso = 0; proceso < hilos; proceso++) {
    int offsetIzq = (proceso == 0) ? 0 : offset;
    int offsetDer = (proceso + 1 == hilos) ? 0 : offset;

    // generar sub imagen a ser procesada por el hilo
    auto regionAprocesar =
        cv::Rect(0, acumulado[proceso] - offsetIzq, imagenRecibida.cols, particiones[proceso] + offsetIzq + offsetDer);
    cv::Mat imagenAProcesar = imagenRecibida(regionAprocesar).clone();

    transformacion(imagenAProcesar, intensidad);

    auto quitarOffset = cv::Rect(0, offsetIzq, imagenAProcesar.cols, imagenAProcesar.rows - offsetIzq - offsetDer);

    imagenesProcesadas[proceso] = imagenAProcesar(quitarOffset);
  }

  for (const auto &imagenProcesada : imagenesProcesadas) {
    nuevaImagen.push_back(imagenProcesada);
  }

  auto quitarOffset = cv::Rect(offsetIzqMPI, 0, nuevaImagen.cols - offsetIzqMPI - offsetDerMPI, nuevaImagen.rows);

  auto imagenRecortada = nuevaImagen(quitarOffset);

  enviarImagenMPI(imagenRecortada, 0);
}

#endif