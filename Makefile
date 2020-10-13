# Generated automatically from Makefile.in by configure.
ALL: default
##### User configurable options #####

SHELL       = /bin/sh
ARCH        = LINUX
COMM        = ch_p4
MPIR_HOME   = /opt/mpich
CC          = mpicxx
CLINKER     = $(CC)
AR          = ar crl
RANLIB      = ranlib
PMPILIB     = -lpmpich
OPTFLAGS    = -O3
MPE_LIBS    = @MPE_LIBS@
MPE_DIR     = /opt/mpich/mpe
MPE_GRAPH   = @MPE_GRAPHICS@
#

### End User configurable options ###

CFLAGS	  = $(OPTFLAGS)
CFLAGSMPE = $(CFLAGS) -I$(MPE_DIR) $(MPE_GRAPH)
CCFLAGS	  = $(CFLAGS)
EXECS	  = tspsec
# bbpar

default: $(EXECS)

all: default

tspsec: tspsec.o libtsp.o
	$(CLINKER) $(OPTFLAGS) -o $@ $^



clean:
	/bin/rm -f *.o $(EXECS)

%.o:	%.cc
	$(CC) $(CFLAGS) -c -o $@ $^

