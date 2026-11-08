# Builds build/libroboracer.so. Default uses the no-op raylib stub in tests/.
# make RENDER=1 fetches raylib 5.5 into .deps/raylib and links it.
CC      ?= cc
# -ffp-contract=off: no compiler-chosen FMA fusion, so the plant's rounding is
# the source's on every compiler and machine (evaluations must reproduce).
CFLAGS   = -std=gnu11 -O3 -march=native -ffp-contract=off -fPIC -fopenmp \
           -Wall -Wextra -Wno-unused -Wno-sign-compare
CORE_MATH = externals/core-math/tanf.c externals/core-math/atanf.c \
            externals/core-math/asinf.c externals/core-math/atan2f.c
SRC      = ocean/roboracer/binding.c ocean/roboracer/renderer.c \
           ocean/roboracer/shim.c $(CORE_MATH)
HDR      = ocean/roboracer/roboracer.h ocean/roboracer/tyre_dr_basin.h \
           ocean/roboracer/renderer.h ocean/roboracer/vecenv.h \
           externals/core-math/core_math.h

RAYLIB_DIR := .deps/raylib
RAYLIB_URL := https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz
RAYLIB_A   := $(RAYLIB_DIR)/lib/libraylib.a

RENDER ?= 0

ifeq ($(RENDER),1)
INC   = -Iocean/roboracer -Iexternals/core-math -I$(RAYLIB_DIR)/include
LIBS  = $(RAYLIB_A) -lGL -lm -lpthread -ldl -lrt -lX11
DEPS  = $(SRC) $(HDR) $(RAYLIB_DIR)/include/raylib.h
else
INC   = -Iocean/roboracer -Iexternals/core-math -Itests
LIBS  = -lm -lpthread
DEPS  = $(SRC) $(HDR) tests/raylib.h
endif

TEST_CFLAGS = -std=gnu11 -O2 -g -Wall -Wextra -Wno-unused -Wno-sign-compare \
              -Iocean/roboracer -Iexternals/core-math -Itests
TEST_RENDERER = ocean/roboracer/renderer.c

# Switching RENDER relinks: the stamp for the other mode is removed, so the
# .so is out of date even when no source changed.
RENDER_STAMP = build/render-$(RENDER).stamp

.PHONY: all test clean
all: build/libroboracer.so

$(RENDER_STAMP):
	@mkdir -p build
	@rm -f build/render-*.stamp
	@touch $@

build/libroboracer.so: $(DEPS) $(RENDER_STAMP)
	$(CC) $(CFLAGS) $(INC) -shared $(SRC) $(LIBS) -o $@

$(RAYLIB_DIR)/include/raylib.h:
	@mkdir -p .deps
	python3 -c "import urllib.request,tarfile,shutil,pathlib;\
d=pathlib.Path('.deps');\
t=pathlib.Path('/tmp/raylib-5.5_linux_amd64.tar.gz');\
urllib.request.urlretrieve('$(RAYLIB_URL)', t);\
tarfile.open(t).extractall(d);\
p=d/'raylib';\
shutil.rmtree(p, ignore_errors=True);\
(d/'raylib-5.5_linux_amd64').rename(p)"

test:
	uv run --frozen python tests/reference.py
	uv run --frozen python tests/reference_edt.py
	$(CC) $(TEST_CFLAGS) tests/test_dynamics.c $(TEST_RENDERER) $(CORE_MATH) -lm -o tests/test_dynamics
	$(CC) $(TEST_CFLAGS) tests/test_env.c $(TEST_RENDERER) $(CORE_MATH) -lm -o tests/test_env
	$(CC) $(TEST_CFLAGS) tests/test_edt.c $(TEST_RENDERER) $(CORE_MATH) -lm -o tests/test_edt
	./tests/test_dynamics
	./tests/test_env
	./tests/test_edt

clean:
	rm -rf build tests/test_dynamics tests/test_env tests/test_edt \
	       tests/fixtures.h tests/fixtures_edt.h
