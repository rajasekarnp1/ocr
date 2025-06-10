# OCR Application Profiling Strategy

**1. Introduction**

The primary purpose of profiling this OCR application is to systematically identify performance bottlenecks. Understanding where the application spends most of its execution time is crucial for guiding optimization efforts. These efforts may include refining algorithms, improving data handling, or strategically expanding the use of high-performance Mojo components to replace or augment existing Python/OpenCV/ONNX Runtime operations.

**2. Key Areas for Profiling**

The following components and operations are anticipated to be the most computationally intensive and therefore key candidates for profiling:

*   **ONNX Model Inference:**
    *   `GeometricCorrector.correct()`: Specifically, the `self.session.run()` call for the geometric correction model.
    *   `ONNXRecognizer.predict()`: Specifically, the `self.session.run()` call for the text recognition model.
*   **Image Preprocessing (Python/OpenCV/NumPy):**
    *   `ImageBinarizer.binarize()`: Particularly OpenCV's Otsu thresholding (`cv2.threshold`) and color conversion (`cv2.cvtColor`).
    *   `ImageDeskewer.deskew()`: Involves `cv2.findContours`, `cv2.minAreaRect`, and `cv2.warpAffine`, which can be intensive.
    *   Image Normalization (Python/NumPy fallback): The `astype(np.float32) / 255.0` operation if the Mojo version is not used or fails.
*   **Mojo Calls (including Python-Mojo interop overhead):**
    *   `normalize_flat_u8_to_float32_mojo` (in `mojo_image_utils.mojo`): The core normalization logic and the overhead of converting data to/from Python lists.
    *   `calculate_histogram_mojo` (in `mojo_image_utils.mojo`): Although not currently integrated into the main orchestrator pipeline, if it were, its performance would be of interest.
    *   `example_mojo_tensor_operation` (in `mojo_recognizer_utils.mojo`): While a simple example, profiling its call can give insights into the baseline overhead of Python-Mojo function calls.
*   **Image Loading and PIL/NumPy Conversions:**
    *   Initial image loading using `PIL.Image.open()` for very large or numerous images.
    *   Conversion from PIL Image objects to NumPy arrays.
*   **Data Conversions for Mojo Interop:**
    *   NumPy array manipulations such as `flatten().tolist()` before calling Mojo functions.
    *   Conversion of results from Mojo (Python lists) back to NumPy arrays (`np.array(...).reshape(...)`).

**3. Profiling Tools (Conceptual)**

*   **Python Profilers:**
    *   **`cProfile` with `pstats` (or `snakeviz` for visualization):**
        *   Description: `cProfile` is a built-in Python profiler that provides deterministic profiling with detailed statistics on call counts and time spent in functions. `pstats` is used for analyzing the output, and `snakeviz` can visualize it.
        *   Application: Ideal for getting an initial overview of the entire pipeline's performance when running `main.py` or specific `OCRWorkflowOrchestrator.process_document()` calls. Helps identify high-level function hotspots.
        *   Example (conceptual):
            ```bash
            python -m cProfile -o profile_output.pstats main.py --image sample_images/sample_medium.png
            # Then in Python:
            # import pstats
            # p = pstats.Stats('profile_output.pstats')
            # p.sort_stats('cumulative').print_stats(20)
            # or using snakeviz:
            # snakeviz profile_output.pstats
            ```
    *   **`line_profiler`:**
        *   Description: Profiles the time spent on individual lines of code within specified functions. Requires decorating target functions with `@profile`.
        *   Application: Excellent for a deep dive into specific Python functions that `cProfile` identifies as bottlenecks (e.g., Python-based image manipulation loops, complex logic in orchestrator or postprocessing).
        *   Example (conceptual):
            ```python
            # In preprocessing_module.py
            # @profile
            # def some_complex_python_func(...):
            #     ...
            ```
            ```bash
            kernprof -l -v script_calling_the_function.py
            ```
    *   **`Scalene`:**
        *   Description: A high-performance CPU and memory profiler for Python. It distinguishes between time spent in Python code, native extensions (C/C++), and system time. It can also profile memory usage, detect copy hotspots, and estimate potential speedups from parallelization.
        *   Application: Highly valuable for this project due to the mix of Python, OpenCV (C++), ONNX Runtime (C++), and Mojo (via CPython interop). Scalene can help pinpoint if bottlenecks are in Python glue code or within the native libraries themselves.
        *   Example (conceptual):
            ```bash
            scalene main.py --image sample_images/sample_large.png
            ```
*   **Mojo Profiling:**
    *   Research: As of my last knowledge update, the Mojo SDK is still under active development. While it has a focus on performance, dedicated, mature profiling tools specifically *within* Mojo (akin to `gprof` for C++ or `pprof` for Go) might still be emerging or have specific methodologies.
    *   Hypothetical Tools/Approaches:
        *   **Built-in Mojo Profiler:** If the Mojo SDK provides a command-line tool or library for profiling Mojo code execution directly (e.g., `mojo profile my_mojo_script.mojo`), that would be the preferred method for Mojo-native parts.
        *   **Timing within Mojo:** Mojo code itself can include timing mechanisms (e.g., using its equivalent of `time.now()` or specific performance counters if available) to measure execution time of specific code blocks or functions and print this data.
        *   **Interop Call Timing:** As currently planned, timing the calls to Mojo functions from Python.
*   **Manual Timing (Python-side):**
    *   Description: Using `time.perf_counter()` in Python before and after calls to specific code blocks or functions.
    *   Application: Useful for quick, targeted measurements of specific operations, especially for:
        *   Calls to Mojo functions (measures total time including interop overhead).
        *   Calls to specific OpenCV functions.
        *   Data conversion steps (e.g., NumPy array to Python list for Mojo).
    *   Example:
        ```python
        # import time
        # start_time = time.perf_counter()
        # mojo_result = mojo_normalize_function(...)
        # end_time = time.perf_counter()
        # logger.debug(f"Mojo function execution time: {end_time - start_time:.6f} seconds")
        ```

**4. Profiling Strategy**

1.  **Baseline Measurement:**
    *   Prepare a diverse set of representative input images: small, medium, large resolutions; simple (text-only) and complex (noisy, varied layout) documents.
    *   Execute the full OCR pipeline using `main.py` for each sample image.
    *   Use `cProfile` (with `pstats`/`snakeviz`) and `Scalene` to capture initial performance profiles. This will provide a high-level view of time distribution across the Python components and native libraries.

2.  **Detailed Analysis (Python):**
    *   Analyze the `cProfile` and `Scalene` outputs to identify Python functions consuming significant CPU time or showing high per-call latency.
    *   Apply `line_profiler` (using `@profile` decorator and `kernprof`) to these identified Python hotspots to understand line-by-line execution costs. This is particularly relevant for custom Python logic in preprocessing, postprocessing, or orchestration.

3.  **Mojo Function Analysis:**
    *   **Interop Call Timing:** In `ocr_workflow_orchestrator.py` (or other relevant Python callers), wrap calls to Mojo functions (e.g., `normalize_flat_u8_to_float32_mojo`) with `time.perf_counter()` to measure the total execution time, including Python-to-Mojo data conversion and call overhead.
    *   **Internal Mojo Timing (if available):** If future Mojo versions offer internal timing utilities or profilers, utilize them to measure execution time purely within the Mojo environment, excluding Python interop overhead.
    *   **Comparison:** Compare the performance of Mojo functions (e.g., normalization) against their Python/NumPy fallback implementations using the same input data to quantify the speedup.
    *   **Data Conversion Overhead:** Profile the Python code responsible for preparing data for Mojo calls (e.g., `np.flatten().tolist()`) and processing data returned from Mojo (e.g., `np.array(...).reshape(...)`) to understand the cost of interop data marshalling.

4.  **Iterative Approach:**
    *   Based on the bottlenecks identified (e.g., a slow Python loop, an inefficient OpenCV call, high data conversion cost for Mojo), implement targeted optimizations.
    *   After each significant optimization, re-run the relevant profilers (`cProfile`, `Scalene`, `line_profiler`, manual timers) with the same sample images to measure the impact of the changes and ensure no new bottlenecks have been introduced.

5.  **Metrics to Collect:**
    *   **Total Pipeline Execution Time:** Overall time from `main.py` invocation to result display.
    *   **Per-Function/Module Time:** Time spent in major Python functions and modules (from `cProfile`, `Scalene`).
    *   **Line-Level Runtimes:** For critical Python functions (from `line_profiler`).
    *   **Call Counts:** Number of times functions are called (from `cProfile`).
    *   **CPU Time Breakdown (Python vs. Native):** Using `Scalene`, differentiate time spent in Python code versus underlying C/C++ libraries (OpenCV, ONNX Runtime) or Mojo.
    *   **Memory Usage:** Peak memory usage and memory allocation patterns (using `Scalene` or other memory profiling tools like `memory_profiler`).
    *   **Mojo Function Execution Time:** Specific timings for Mojo function calls, including interop overhead (from manual timing or Mojo-specific tools).

**5. Reporting**

*   Compile a summary of profiling findings, clearly highlighting the top 2-3 performance bottlenecks in the application with quantitative data (e.g., "Function X accounts for 40% of total execution time").
*   Provide specific, actionable recommendations for optimization. These might include:
    *   Porting specific CPU-bound Python functions (identified by profilers) to Mojo.
    *   Optimizing existing Mojo functions (e.g., exploring SIMD or other advanced Mojo features if initial Mojo implementations are not meeting performance targets).
    *   Refining Python data handling and reducing data conversion overhead, especially around calls to native libraries or Mojo.
    *   Optimizing image processing parameters or algorithms in OpenCV.
    *   Investigating ONNX Runtime execution providers or model optimization if inference is a major bottleneck.
*   The report should guide decisions on where to focus development efforts for the most significant performance gains.

This profiling strategy aims to provide a structured approach to understanding and improving the performance of the OCR application.
