#ifndef Particionado_H
#define Particionado_H

#include <vector>

/**
 * Funcion que genera particiones de un numero, soporta divisiones no exactas
 * agregando el resto de la division a las particiones.
 * Por este motivo las particiones no necesariamente son iguales entre si.
 * Ej. 100 -> 3 -> [34,33,33]
 *
 * @param cantidad numero total de elementos a particiones.
 * @param numeroParticiones numero de particiones resultante.
 * @returns vector que contiene el nuemro de elementos en cada particion.
 */
std::vector<int> particionar(int cantidad, int numeroParticiones) {
  int particion = cantidad / numeroParticiones;
  std::vector<int> particiones(numeroParticiones, particion);

  auto resto = cantidad - particion * numeroParticiones;
  for (auto &particion : particiones) {
    if (resto--)
      particion++;
    else
      break;
  }
  return particiones;
}

/**
 * Funcion que genera un vector con la acumulación parcial de otro vector.
 * Usado para calcular rangos de subdivisiones.
 * Ej. [34,33,33] -> [0,34,67]
 *
 * @param particiones vector que contiene las cantidades.
 * @returns vector que contiene el nuemro de elementos acumulados hasta esa posicion.
 */
std::vector<int> vectorAcumulador(const std::vector<int> particiones) {
  std::vector<int> acumulado(particiones.size());
  acumulado[0] = 0;

  for (size_t i = 1; i < particiones.size(); i++) {
    acumulado[i] = acumulado[i - 1] + particiones[i];
  }
  return acumulado;
}

#endif