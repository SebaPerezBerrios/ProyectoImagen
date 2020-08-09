#!/bin/bash

mpirun --hostfile maquinas.txt ./dist/programa 2 test.png
