# Hexagon DSP ORB

Hexagon ORB is an optional Texture backend. It is disabled by default. Descriptor type 4 selects CPU ORB, type 5 selects CUDA ORB when built with CUDA, and type 6 selects the DSP backend.

The host prepares the image region. The DSP performs ORB extraction and batched Hamming 2-NN matching through FastRPC; geometric filtering and the M3T optimizer remain on the host. DSP errors fail the texture operation rather than silently switching backends.

## Build

Provide a Hexagon skeleton library and an ARM FastRPC stub library, then configure from the repository root:

    cmake -S . -B target/build-dsp -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DCMAKE_PREFIX_PATH="$RBT_PREFIX" \
      -DCMAKE_INSTALL_PREFIX="$PWD/target/m3t-dsp-install" \
      -DRBT_M3T_UPSTREAM_ROOT=/path/to/target/m3t-upstream \
      -DRBT_M3T_ORBBEC_SOURCE_DIR=/path/to/rbt_robot_ws/drivers/orbbec \
      -DRBT_M3T_ENABLE_HEXAGON_ORB=ON \
      -DRBT_M3T_DSP_SKEL=/abs/path/librbt_m3t_orb_skel.so \
      -DRBT_M3T_DSP_STUB=/abs/path/librbt_m3t_orb_stub.so
    cmake --build target/build-dsp --parallel 4
    cmake --install target/build-dsp

Set RBT_M3T_DSP_MODULE and RBT_M3T_DSP_STUB to override the installed runtime libraries. Set RBT_M3T_DSP_AUDIT=1 to emit backend, extraction, and matching events.

The backend uses a deterministic fixed-point implementation. Device tests matched the portable CPU core byte-for-byte on the tested cases; OpenCV ORB is not bitwise compatible. V73 measurements showed the current DSP facade slower than the tested CPU paths for extraction and matching.

Supported inputs are up to 1024×1024 pixels, 2048 features, 1–8 pyramid levels, and scale factors from 1.05 to 2.0. WTA_K is 2, patch size and edge threshold are 31, and FAST threshold is 20. DSP workspace is limited to 16 MiB and must be 128-byte aligned.

See [BUILD.md](BUILD.md) for the Hexagon toolchain steps and [PROVENANCE.md](PROVENANCE.md) for generated-code sources and licenses.
