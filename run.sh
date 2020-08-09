#!/bin/bash

mpirun --hostfile maquinas.txt ./dist/programa 1 test.png
