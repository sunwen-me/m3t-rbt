# Hexagon build

scripts/build_hexagon.sh builds the DSP module and ARM FastRPC stub from the checked-in QAIC interface. Generated interface files are used by default; set QAIC_REGENERATE=1 to regenerate them. Write build outputs outside the source tree.

Required tools are a Hexagon compiler, Qualcomm ELD, an ARM64 host compiler, and the target FastRPC library. See PROVENANCE.md for the tested toolchain and pinned inputs.

When the Hexagon compiler and ELD are on the same host:

    HEXAGON_CC=/opt/hexagon-sdk/bin/hexagon-clang \
    HEXAGON_CXX=/opt/hexagon-sdk/bin/hexagon-clang++ \
    HEXAGON_LINK=/opt/eld/bin/hexagon-link \
    HEXAGON_MCPU=hexagonv68 \
    HOST_CC=aarch64-linux-gnu-gcc \
    FASTRPC_LIBS='-L/opt/aidlux/lib -lcdsprpc' \
    ./dsp/scripts/build_hexagon.sh \
      --phase all --output-dir "$PWD/target/m3t-dsp-hexagon"

If ELD runs on a separate host, run compile and link phases separately, copying the output directory between hosts:

    HEXAGON_CC=/opt/hexagon-sdk/bin/hexagon-clang \
    HEXAGON_CXX=/opt/hexagon-sdk/bin/hexagon-clang++ \
    HEXAGON_MCPU=hexagonv68 HOST_CC=aarch64-linux-gnu-gcc \
    FASTRPC_LIBS='-L/opt/aidlux/lib -lcdsprpc' \
    ./dsp/scripts/build_hexagon.sh \
      --phase compile --output-dir "$PWD/target/m3t-dsp-hexagon"

    HEXAGON_LINK=/opt/eld/bin/hexagon-link \
    ./dsp/scripts/build_hexagon.sh \
      --phase link --output-dir "$PWD/target/m3t-dsp-hexagon"

The compile phase targets hexagon-unknown-none-elf with C++17. It disables exceptions, RTTI, fast-math, and FP contraction. Set HEXAGON_CPPFLAGS and HEXAGON_CFLAGS for SDK-specific headers and options. Set HOST_CPPFLAGS, HOST_CFLAGS, HOST_LDFLAGS, and FASTRPC_LIBS for the ARM64 stub toolchain.

The link phase checks the DSP ELF identity, DT_HEXAGON_VER=3, load segments, unresolved symbols, dynamic dependencies, and the exported QAIC invoke entry point.

Run the device test with absolute library paths:

    RBT_M3T_DSP_AUDIT=1 \
    target/build-dsp/rbt_m3t_dsp_device_test \
      /abs/path/librbt_m3t_orb_stub.so \
      /abs/path/librbt_m3t_orb_skel.so
