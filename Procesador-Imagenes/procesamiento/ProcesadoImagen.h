#ifndef ProcesadoImagen_H
#define ProcesadoImagen_H

#include <opencv2/imgproc.hpp>

void escalarImagen(cv::Mat &imagen) {
  // Se elije interpolacion lineal a escala 1.33 para redimensionar la imagen
  resize(imagen, imagen, cv::Size(), 1.33, 1.33);
}

void escalaGrisesImagen(cv::Mat &imagen) { cv::cvtColor(imagen, imagen, cv::COLOR_BGR2GRAY); }

void difuminarImagen(cv::Mat &imagen, int intensidad) {
  // se elije difuminado de caja, entrega mejor rendimiento que el dif gaussiano con una calidad similar.
  // para mejorar la aproximacion al difuminado gaussiano se aplica dos veces
  blur(imagen, imagen, cv::Size(intensidad, intensidad), cv::Point(-1, -1));
  blur(imagen, imagen, cv::Size(intensidad, intensidad), cv::Point(-1, -1));
}

#endif