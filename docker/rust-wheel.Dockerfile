FROM quay.io/pypa/manylinux2014_x86_64

ENV RUSTUP_HOME=/opt/aimdo-rustup \
    CARGO_HOME=/opt/aimdo-cargo
ENV PATH=/opt/aimdo-cargo/bin:${PATH}
ENV AIMDO_BUILD_PYTHON=/opt/python/cp310-cp310/bin/python

RUN curl -fsSL https://sh.rustup.rs -o /tmp/aimdo-rustup.sh \
    && sh /tmp/aimdo-rustup.sh -y --profile minimal --default-toolchain 1.85.1 \
    && rm /tmp/aimdo-rustup.sh \
    && "$AIMDO_BUILD_PYTHON" -m pip install 'setuptools>=77'

WORKDIR /src
CMD ["bash", "scripts/build-rust-wheel.sh"]
