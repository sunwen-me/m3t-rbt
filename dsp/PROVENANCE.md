# DSP dependency provenance

The DSP implementation and IDL are maintained in this repository. Build tools and runtime libraries are supplied by the build and target environments.

## QAIC

- Repository: https://github.com/qualcomm/QAIC
- Source commit: a66a0019fb488abae955bbc9b3a9de3dc15962d0
- Tested generator: QAIC 01.00.51
- Generated header, skeleton, and AP stub: [generated/qaic](generated/qaic/)
- Source archive SHA-256: 7f033a57adee248962a4317b8b4367060bdef1e7fb50fac403e7e21e2552f5af
- QAIC license and notice: [licenses/qaic-LICENSE.txt](licenses/qaic-LICENSE.txt), [licenses/qaic-NOTICE](licenses/qaic-NOTICE)

The IDL dependency include/fastrpc/remote.idl is from the same commit and retains its BSD-3-Clause-Clear notice.

## FastRPC headers

- Repository: https://github.com/quic/fastrpc
- Source commit: 75e71493a8fbe84ae84db153b6f9cefca8f9e19a
- Copied public headers are in include/fastrpc/.
- License: [licenses/fastrpc-LICENSE.txt](licenses/fastrpc-LICENSE.txt), SHA-256 683e80f5436e6b0761139e626d2c764870ea0ed19b14916ef4c82772b99c4d44

## compiler-rt helpers

The DSP build includes sfdiv_opt.S and udivsi3.S from LLVM compiler-rt at llvmorg-15.0.7. Both retain Apache-2.0 WITH LLVM-exception notices; see [licenses/compiler-rt-LICENSE.TXT](licenses/compiler-rt-LICENSE.TXT).

## Tested toolchain

The device build used Ubuntu Clang 15.0.7 for Hexagon v68, Qualcomm ELD 24.0, ARM64 GCC 11.4.0, and the target's FastRPC runtime on a V73 CDSP. These versions describe the tested build; use the matching headers and libraries for the target SDK. The source package does not include the compiler, linker, SDK, or FastRPC binary.
