#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "lib/Utils.h"
#include "procesamiento/ProcesadoImagen.h"
#include "procesamiento/TransporteImagen.h"
#include "procesamiento/TransporteImagenOffset.h"

#define Orquestador 0

int main(int argc, char **argv) {
  int mi_rango;
  int procesosTotales;
  int procesosReservados = 1;
  int procesosMinimos = 2;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);

  MPI_Comm_size(MPI_COMM_WORLD, &procesosTotales);
  if (procesosTotales < procesosMinimos) {
    std::cerr << "ERROR, el programa necesita al menos 2 nodos de procesamiento" << std::endl;
    return EXIT_FAILURE;
  }

  if (argc < 2) {
    std::cerr << "Se nececita una operacion y un nombre de imagen" << std::endl;
    return EXIT_FAILURE;
  }
  if (argc < 3) {
    std::cerr << "Se nececita un nombre de imagen" << std::endl;
    return EXIT_FAILURE;
  }

  auto tipoProceso = std::string(argv[1]);

  if (mi_rango == Orquestador) {
    // lectura imagen, por defecto se convierte a 4 canales para compatibilidad con archivos PNG con transparencia
    std::string nombreImagen;
    try {
      nombreImagen = cv::samples::findFile(argv[2]);
    } catch (...) {
      std::cerr << "Imagen no encontrada" << std::endl;
      return EXIT_FAILURE;
    }

    cv::Mat imagenOriginal = cv::imread(nombreImagen, cv::IMREAD_UNCHANGED);

    if (tipoProceso == "1") {
      // se calcula el offset necesario para el difuminado
      auto offset = calculoOffset(imagenOriginal);
      enviarImagenOffset(procesosReservados, procesosTotales, offset, imagenOriginal);
    } else if (tipoProceso == "2") {
      enviarImagen(procesosReservados, procesosTotales, imagenOriginal);
    } else if (tipoProceso == "3") {
      enviarImagen(procesosReservados, procesosTotales, imagenOriginal);
    } else {
      std::cerr << "Opcion no valida" << std::endl;
      return EXIT_FAILURE;
    }

    auto imagenRecibida = unirImagen(procesosReservados, procesosTotales);

    std::stringstream nombreArchivo;
    nombreArchivo << std::setfill('0');
    nombreArchivo << "operacion_" << tipoProceso << "_" << obtenerTiempo() << ".png";
    imwrite(nombreArchivo.str(), imagenRecibida);

    participante();
  }

  if (mi_rango != Orquestador) { /* Esclavo */
    if (tipoProceso == "1") procesarImagenOffset(difuminarImagen);
    if (tipoProceso == "2") procesarImagen(escalaGrisesImagen);
    if (tipoProceso == "3") procesarImagen(escalarImagen);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
