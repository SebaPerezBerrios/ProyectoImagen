#ifndef Utils_H
#define Utils_H

#include <ctime>
#include <iomanip>
#include <opencv2/core.hpp>

#include "FuncionesMPI.h"

cv::Mat unirImagen(int procesosReservados, int procesosTotales) {
  auto imagenGenerada = cv::Mat();
  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear nueva imagen a partir de particiones
  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    auto imagenRecibida = recibirImagenMPI(procesosReservados + proceso);
    imagenGenerada.push_back(imagenRecibida);
  }

  return imagenGenerada;
}

std::string obtenerTiempo() {
  auto tiempo = time(0);
  auto tiempoLocalPtr = std::localtime(&tiempo);
  std::stringstream tiempoLocal;
  tiempoLocal << std::setfill('0');
  tiempoLocal << std::setw(4) << tiempoLocalPtr->tm_year + 1900 << std::setw(2) << tiempoLocalPtr->tm_mon + 1
              << std::setw(2) << tiempoLocalPtr->tm_mday << std::setw(2) << tiempoLocalPtr->tm_hour << std::setw(2)
              << tiempoLocalPtr->tm_min << std::setw(2) << tiempoLocalPtr->tm_sec;
  return tiempoLocal.str();
}

int calculoOffset(const cv::Mat &imagen) {
  int ladoMasCorto = std::min(imagen.rows, imagen.cols);
  int difuminado = (double)ladoMasCorto * 0.02;
  return (difuminado % 2 == 0) ? difuminado : difuminado + 1;
}

void participante() {
  std::cout << std::endl << "=== Trabajo tratamiento de imagenes ===" << std::endl;
  std::cout << std::endl << "Sebastián Pérez Berrios" << std::endl;
  std::cout << std::endl << "Ivan Pérez" << std::endl;
  std::cout << std::endl << "Lester Vasquez" << std::endl;
}

#endif