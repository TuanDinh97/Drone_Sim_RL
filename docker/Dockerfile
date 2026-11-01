# quadsim on the official Isaac Lab container (Isaac Sim 5.1.0 + Isaac Lab v2.3.2, Ubuntu 22.04, Python 3.11).
# Build:  docker compose -f docker/docker-compose.yaml build
# The base image already contains torch 2.7.0+cu128, rsl-rl-lib, skrl, warp; we only add our package.
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${BASE_IMAGE}

ENV ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    QUADSIM_ROOT=/workspace/quadsim \
    PYTHONUNBUFFERED=1

# make `python` resolve to the Isaac Sim interpreter (python.sh must be exec'd from its own directory tree)
RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python \
 && chmod +x /usr/local/bin/python && ln -sf /usr/local/bin/python /usr/local/bin/python3

COPY docker/requirements.txt /tmp/quadsim-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/quadsim-requirements.txt

WORKDIR ${QUADSIM_ROOT}
COPY source/quadsim ${QUADSIM_ROOT}/source/quadsim
RUN python -m pip install --no-cache-dir --no-deps -e ${QUADSIM_ROOT}/source/quadsim

# scripts/tests/docs are bind-mounted by docker-compose for development; copy them for standalone use
COPY scripts ${QUADSIM_ROOT}/scripts
COPY tests ${QUADSIM_ROOT}/tests
COPY examples ${QUADSIM_ROOT}/examples
COPY env ${QUADSIM_ROOT}/env

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["python -m pytest tests/core -q"]
