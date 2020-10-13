/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Paralelo                    */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libtsp.h"

#define BUFFER_SIZE 1000

using namespace std;

unsigned int NCIUDADES;
char *input_file;
 
int main (int argc, char **argv) {
    
    MPI::Init(argc,argv);

    switch (argc) {
		case 3:
			NCIUDADES = atoi(argv[1]);
			break;
		default:
			cerr << "La sintaxis es: bbseq <tamaño> <archivo>" << endl;
			exit(1);
			break;
	}

	bool activo, nueva_U, balanceoInicial;
	char buff[BUFFER_SIZE], buffSol[100][BUFFER_SIZE];
	int rank, size, U, i, j, flag, position, signal, matriz[NCIUDADES*NCIUDADES], **tsp0 = reservarMatrizCuadrada(NCIUDADES); // NCIUDADES = orden de la matriz = nº elementos //sempre seran matrices cuadradas
	double t;
	tNodo nodo, lnodo, rnodo, solucion; 
	tPila pila;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Request request, requestWork[size], requestSol[size];

	U = INFINITO;
	PilaInic(&pila);
	position = 0;
	signal = 1;

	if(rank == 0) {
		InicNodo(&nodo);
		LeerMatriz(argv[2], tsp0);

		t = MPI::Wtime();
		balanceoInicial = false;
		if(!(activo = !Inconsistente(tsp0))) {
			cerr << "La matriz leida es inconsistente, revisa los valores de entrada" << endl;
			exit(2);
		}

		printf("Iniciamos ...\n");
		for(i = 0; i<NCIUDADES; i++) for (j = 0; j<NCIUDADES; matriz[position] = tsp0[i][j], j++, position++) {}
		MPI_Bcast(&matriz, NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

		for(i = 1; i<size; MPI_Irecv(&signal, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &requestWork[i]), i++) {}
	}
	else {
		MPI_Bcast(&matriz, NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
		for(i = 0; i<NCIUDADES; i++) for (j = 0; j<NCIUDADES; tsp0[i][j] = matriz[position], j++, position++) {}
		MPI_Send(&signal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD); // envia int petición de trabajo
		MPI_Recv(&buff, (3+NCIUDADES+NCIUDADES-2)*4, MPI_PACKED, 0, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // recepción de trabajo, en cuanto el padre pueda assignar se completarà esta acción
		position = 0;
		MPI_Unpack(buff, BUFFER_SIZE, &position, &nodo.ci, 1, MPI_INT, MPI_COMM_WORLD); // desenpaquetamos el primer entero
		if(nodo.ci != -1) { // verificamos que tenemos trabajo
			MPI_Unpack(buff, BUFFER_SIZE, &position, nodo.incl, NCIUDADES, MPI_INT, MPI_COMM_WORLD);
			MPI_Unpack(buff, BUFFER_SIZE, &position, &nodo.orig_excl, 1, MPI_INT, MPI_COMM_WORLD);
			MPI_Unpack(buff, BUFFER_SIZE, &position, nodo.dest_excl, NCIUDADES-2, MPI_INT, MPI_COMM_WORLD);
			activo = balanceoInicial = true;
		} else {
			activo = false;
		}
	}

	for(i = 0; i<size; i++) { 	// inicio recepcion soluciones parciales
		if(rank != i) {
			MPI_Irecv(&buffSol[i], (2+NCIUDADES+NCIUDADES-2)*4, MPI_PACKED, i, 7, MPI_COMM_WORLD, &requestSol[i]);
		}
	}

	while (activo) {
		Ramifica (&nodo, &lnodo, &rnodo, tsp0);
		nueva_U = false;
		if (Solucion(&rnodo)) {
			if (rnodo.ci < U) {
				U = rnodo.ci;
				nueva_U = true;
				CopiaNodo(&rnodo, &solucion);
			}
		}
		else {
			if (rnodo.ci < U) {
				if (!PilaPush (&pila, &rnodo)) {
					printf("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}
		if (Solucion(&lnodo)) {
			if (lnodo.ci < U) {
				U = lnodo.ci;
				nueva_U = true;
				CopiaNodo(&lnodo,&solucion);
			}
		}
		else {
			if (lnodo.ci < U) {
				if (!PilaPush (&pila, &lnodo)) {
					printf("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

		if (nueva_U) { // se ha encontrado una solución nueva -> comunicación global a todos los procesos
			if(balanceoInicial) { // si hay balanceoInicial entre procesos puedo enviar la solución
				position = 0;
				MPI_Pack(&solucion.ci, 1, MPI_INT, buffSol, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				MPI_Pack(solucion.incl, NCIUDADES, MPI_INT, buffSol, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				MPI_Pack(&solucion.orig_excl, 1, MPI_INT, buffSol, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				MPI_Pack(solucion.dest_excl, NCIUDADES-2, MPI_INT, buffSol, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				for(i = 0; i<size; i++){
					if(rank != i) {
						MPI_Send(&buffSol[0], position, MPI_PACKED, i, 7, MPI_COMM_WORLD); // solución enviada
					}
				}
			}
			PilaAcotar (&pila, U);
			printf("Proceso %d solución ci encontrada: %d\n", rank, solucion.ci);
		}

		for(i = 0; i<size && balanceoInicial; i++) {
			if(rank != i) {
				MPI_Test(&requestSol[i], &flag, MPI_STATUS_IGNORE); // si algú li ha enviat una solució
				if(flag) {
					position = 0;
					MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, &nodo.ci, 1, MPI_INT, MPI_COMM_WORLD);
					if (nodo.ci < U) { // se ha encontrado una solucion mejor
						U = nodo.ci;
						MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, nodo.incl, NCIUDADES, MPI_INT, MPI_COMM_WORLD);
						MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, &nodo.orig_excl, 1, MPI_INT, MPI_COMM_WORLD);
						MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, nodo.dest_excl, NCIUDADES-2, MPI_INT, MPI_COMM_WORLD);
						CopiaNodo(&nodo, &solucion);
						PilaAcotar(&pila, U);
						nueva_U = false;
					}
					MPI_Irecv(&buffSol[i], (2+NCIUDADES+NCIUDADES-2)*4, MPI_PACKED, i, 7, MPI_COMM_WORLD, &requestSol[i]);
				}
			}
		}

		if(rank == 0 && !balanceoInicial && size>1 && pila.tope >= size) { // balanceo inicial equitativo, un nodo para cada proceso
			balanceoInicial = true;
			for(i = 1; i<size; i++) {
				MPI_Wait(&requestWork[i], MPI_STATUS_IGNORE);
				PilaPop(&pila, &nodo);
				position = 0;
				MPI_Pack(&nodo.ci, 1, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				MPI_Pack(nodo.incl, NCIUDADES, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				MPI_Pack(&nodo.orig_excl, 1, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				MPI_Pack(nodo.dest_excl, NCIUDADES-2, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
				MPI_Send(&buff, position, MPI_PACKED, i, 8, MPI_COMM_WORLD);
				MPI_Irecv(&signal, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &requestWork[i]); // el padre hace los mpi_irecv de requestwork.
			}
		} 
		else { // despues de realizar el balanceo inicial se empieza el envio trabajo a los workers
			if(rank == 0 && pila.tope >= 2 && balanceoInicial) {
				for(i = 1; pila.tope >= 2 && i<size; i++) {
					MPI_Test(&requestWork[i], &flag, MPI_STATUS_IGNORE);
					if(flag) {
						PilaPop(&pila, &nodo);
						position = 0;
						MPI_Pack(&nodo.ci, 1, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
						MPI_Pack(nodo.incl, NCIUDADES, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
						MPI_Pack(&nodo.orig_excl, 1, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
						MPI_Pack(nodo.dest_excl, NCIUDADES-2, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
						MPI_Send(&buff, position, MPI_PACKED, i, 8, MPI_COMM_WORLD);
						MPI_Irecv(&signal, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &requestWork[i]);
					}
				}
			}
		}

		activo = PilaPop (&pila, &nodo);
		if(rank != 0 && activo == false) { // solicitar trabajo al padre ya que no me quedan nodos en la pila
			MPI_Send(&signal, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
			MPI_Recv(&buff, (2+NCIUDADES+NCIUDADES-2)*4, MPI_PACKED, 0, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // los workers esperan a tener trabajo

			position = 0;
			MPI_Unpack(buff, BUFFER_SIZE, &position, &nodo.ci, 1, MPI_INT, MPI_COMM_WORLD);
			if(nodo.ci != -1) {
				MPI_Unpack(buff, BUFFER_SIZE, &position, nodo.incl, NCIUDADES, MPI_INT, MPI_COMM_WORLD);
				MPI_Unpack(buff, BUFFER_SIZE, &position, &nodo.orig_excl, 1, MPI_INT, MPI_COMM_WORLD);
				MPI_Unpack(buff, BUFFER_SIZE, &position, nodo.dest_excl, NCIUDADES-2, MPI_INT, MPI_COMM_WORLD);
				activo = true;
			}
		}
	}

	if(rank==0) {
		j = 0;
		while(j < size-1) {
			for(i = 1; i<size; i++) {
				if(MPI_Test(&requestSol[i], &flag, MPI_STATUS_IGNORE) == MPI_SUCCESS && flag) {
					position = 0;
					MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, &nodo.ci, 1, MPI_INT, MPI_COMM_WORLD);
					if (nodo.ci > 0 && nodo.ci < U) { // se ha encontrado una solucion mejor
						U = nodo.ci;
						MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, nodo.incl, NCIUDADES, MPI_INT, MPI_COMM_WORLD);
						MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, &nodo.orig_excl, 1, MPI_INT, MPI_COMM_WORLD);
						MPI_Unpack(buffSol[i], BUFFER_SIZE, &position, nodo.dest_excl, NCIUDADES-2, MPI_INT, MPI_COMM_WORLD);
						CopiaNodo(&nodo,&solucion);
					}
					MPI_Irecv(&buffSol[i], (2+NCIUDADES+NCIUDADES-2)*4, MPI_PACKED, i, 7, MPI_COMM_WORLD, &requestSol[i]);
				}
				if(MPI_Test(&requestWork[i], &flag, MPI_STATUS_IGNORE) == MPI_SUCCESS && flag) {
					position = 0;
					signal = -1;
					MPI_Wait(&requestWork[i], MPI_STATUS_IGNORE);
					MPI_Pack(&signal, 1, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
					MPI_Send(&buff, position, MPI_PACKED, i, 8, MPI_COMM_WORLD);
					MPI_Irecv(&signal, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &requestWork[i]);
					j++;
					printf("Señal finalización enviada a: %d\n", i);
				}
			}
		}

		t = MPI::Wtime()-t;
		printf ("-------------------------------------------------------------\n");
		printf("Coste Óptimo (ci): %d >> %s.sol\nNúmero de procesos: %d\nTiempo Requerido: %lfs\n", solucion.ci, input_file, size, t);
		printf ("-------------------------------------------------------------\n");
		EscribeSolucion(&solucion, t);
	}

	MPI_Finalize();
	printf("El trabajo %d ha finalizado.\n", rank);
	liberarMatriz(tsp0);
	return 0;
}