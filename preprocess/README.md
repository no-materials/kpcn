This directory contains scripts which are part of the data preprocessing pipeline. Specifically:
* `preprocess_complete_pc.py` is used to generate **complete** point clouds via uniform sampling from a list of models using multi-threading. Uses code from the `sample` package.
* `preprocess_partial_pc.py` is used to generate **partial** point clouds via simulating a virtual depth scan using Blender from a list of models using multi-threading. Uses code from the `render` package.

Please check the `README.md` within each package (`render` & `sample`) in order to setup and build necessary dependencies.