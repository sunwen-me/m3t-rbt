# Verification

Use the SDK, upstream checkout, and Orbbec driver paths from the build section in README.md. Configure with BUILD_TESTING enabled, build, then run the native tests:

    cmake -S . -B target/build -DBUILD_TESTING=ON
    cmake --build target/build --parallel 4
    ctest --test-dir target/build --output-on-failure

The suite covers frame input, model construction, configuration validation, camera motion, joint priors, tracking output and TF, and the portable ORB core.

Run the model generator tests with:

    python3 -m unittest discover \
      -s models/so101_lerobot_soft_fin -p 'test_build_model.py' -v

Fixture and device runners are under scripts/ and tests/. Device-dependent checks require the matching RBT SDK, camera provider, renderer, or DSP libraries.
