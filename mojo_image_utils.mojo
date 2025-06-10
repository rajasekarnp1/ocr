from python import PythonObject, Python, List, FloatLiteral, IntLiteral

fn normalize_flat_u8_to_float32_mojo(flat_u8_list_obj: PythonObject, height: Int, width: Int) -> PythonObject:
    """
    Normalizes a flat list of UInt8 image pixel values to Float32 values (pixel / 255.0).

    Args:
        flat_u8_list_obj: A PythonObject representing a Python list of uint8 pixel values.
        height: The height of the original image.
        width: The width of the original image.

    Returns:
        A PythonObject representing a Python list of float32 normalized pixel values.
    """
    print("[Mojo] normalize_flat_u8_to_float32_mojo called. Expected flat list length:", height * width)

    try:
        # Convert PythonObject to Mojo List of IntLiteral (as Mojo's List[UInt8] from PythonObject might be tricky)
        # Python side sends list of ints (0-255 range)
        u8_input_list = Python.List[IntLiteral](flat_u8_list_obj)
        let num_pixels = len(u8_input_list)
        print("[Mojo] Received flat list of length:", num_pixels)

        if num_pixels != height * width:
            print("[Mojo] Error: Unexpected number of pixels. Expected", height * width, ", got", num_pixels)
            # Return an empty list or handle error appropriately
            return Python.List[FloatLiteral]().to_object()

        var float32_output_list = Python.List[FloatLiteral]()

        for i in range(num_pixels):
            let u8_val = u8_input_list[i] # This is an IntLiteral
            let float_val = Float32(u8_val) / 255.0
            float32_output_list.append(float_val)

        print("[Mojo] Normalization complete. Output list length:", len(float32_output_list))
        return float32_output_list.to_object()

    except e:
        print("[Mojo] Error in normalize_flat_u8_to_float32_mojo:", e)
        # Return an empty list to indicate failure
        return Python.List[FloatLiteral]().to_object()


fn main():
    # Example of how to test this function from Mojo (conceptually)
    print("[Mojo] Main function in mojo_image_utils.mojo called for testing.")

    # Create a dummy Python list of IntLiterals (simulating flattened uint8 image data)
    let test_height: Int = 2
    let test_width: Int = 3
    var dummy_flat_u8_pylist = Python.List[IntLiteral]()
    dummy_flat_u8_pylist.append(0)
    dummy_flat_u8_pylist.append(128)
    dummy_flat_u8_pylist.append(255)
    dummy_flat_u8_pylist.append(50)
    dummy_flat_u8_pylist.append(100)
    dummy_flat_u8_pylist.append(150)

    print("[Mojo] Test input flat list (IntLiteral):", dummy_flat_u8_pylist)

    let result_obj = normalize_flat_u8_to_float32_mojo(dummy_flat_u8_pylist.to_object(), test_height, test_width)

    let result_pylist = Python.List[FloatLiteral](result_obj)
    print("[Mojo] Test output flat list (FloatLiteral) from function:", result_pylist)

    # Expected output: [0.0, 128/255.0, 1.0, 50/255.0, 100/255.0, 150/255.0]
    for i in range(len(result_pylist)):
        print("[Mojo] Normalize test - Result element", i, ":", result_pylist[i])

    print("[Mojo] --- Testing calculate_histogram_mojo ---")
    # Re-use dummy_flat_u8_pylist for histogram test
    let hist_obj = calculate_histogram_mojo(dummy_flat_u8_pylist.to_object(), test_height, test_width)
    let hist_pylist = Python.List[IntLiteral](hist_obj)
    print("[Mojo] Histogram test - Result list from function:", hist_pylist)
    # Expected: bins for 0, 50, 100, 128, 150, 255 should be 1, others 0.
    # Print non-zero bins for brevity
    for i in range(len(hist_pylist)):
        if hist_pylist[i] > 0:
            print("[Mojo] Histogram test - Bin", i, ":", hist_pylist[i])

    print("[Mojo] --- Testing calculate_bounding_box_mojo ---")
    # Create a specific test image for bounding box:
    # 0  0  0  0
    # 0 50  0  0
    # 0  0 80  0
    # 0  0  0  0
    # Expected bbox: (min_r=1, min_c=1, max_r=2, max_c=2)
    let bbox_height: Int = 4
    let bbox_width: Int = 4
    var bbox_test_pylist = Python.List[IntLiteral]()
    bbox_test_pylist.append(0); bbox_test_pylist.append(0); bbox_test_pylist.append(0); bbox_test_pylist.append(0);
    bbox_test_pylist.append(0); bbox_test_pylist.append(50); bbox_test_pylist.append(0); bbox_test_pylist.append(0);
    bbox_test_pylist.append(0); bbox_test_pylist.append(0); bbox_test_pylist.append(80); bbox_test_pylist.append(0);
    bbox_test_pylist.append(0); bbox_test_pylist.append(0); bbox_test_pylist.append(0); bbox_test_pylist.append(0);

    print("[Mojo] BBox test - Input flat list:", bbox_test_pylist)
    let bbox_obj = calculate_bounding_box_mojo(bbox_test_pylist.to_object(), bbox_height, bbox_width)

    # Assuming it returns a Python tuple (or None)
    # For testing, we'll rely on the print statements within the function if it's None
    # If it's a tuple, printing the PythonObject directly is fine for this test.
    print("[Mojo] BBox test - Result PythonObject:", bbox_obj)

    # Test with all zeros (no foreground)
    var bbox_all_zeros_pylist = Python.List[IntLiteral]()
    for _ in range(bbox_height * bbox_width):
        bbox_all_zeros_pylist.append(0)
    print("[Mojo] BBox test - Input all zeros list:", bbox_all_zeros_pylist)
    let bbox_no_fg_obj = calculate_bounding_box_mojo(bbox_all_zeros_pylist.to_object(), bbox_height, bbox_width)
    print("[Mojo] BBox test - Result for no foreground:", bbox_no_fg_obj)


# fn for histogram calculation
fn calculate_histogram_mojo(flat_u8_list_obj: PythonObject, height: Int, width: Int) -> PythonObject:
    """
    Calculates the histogram for a flat list of UInt8 image pixel values.

    Args:
        flat_u8_list_obj: A PythonObject representing a Python list of uint8 pixel values.
        height: The height of the original image.
        width: The width of the original image.

    Returns:
        A PythonObject representing a Python list of 256 integers (the histogram).
    """
    print("[Mojo] calculate_histogram_mojo called. Expected flat list length:", height * width)

    try:
        # Convert PythonObject to Mojo List of IntLiteral
        u8_input_list = Python.List[IntLiteral](flat_u8_list_obj)
        let num_pixels = len(u8_input_list)
        print("[Mojo] Received flat list for histogram of length:", num_pixels)

        if num_pixels != height * width:
            print("[Mojo] Error: Unexpected number of pixels for histogram. Expected", height * width, ", got", num_pixels)
            return Python.List[IntLiteral]().to_object() # Return empty list on error

        # Initialize histogram: Array of 256 Ints, all zeros.
        # Using Python.List for direct return to Python.
        # For pure Mojo, `var histogram_bins = Array[Int, 256]()` might be used, then convert.
        var histogram_pylist = Python.List[IntLiteral]()
        for _ in range(256):
            histogram_pylist.append(0)

        for i in range(num_pixels):
            let pixel_value_literal = u8_input_list[i]
            let pixel_value = Int(pixel_value_literal) # Convert IntLiteral to Int

            if pixel_value >= 0 and pixel_value < 256:
                let current_count = histogram_pylist[pixel_value]
                histogram_pylist[pixel_value] = current_count + 1
            else:
                # This case should ideally not happen if input is truly uint8
                print("[Mojo] Warning: Pixel value", pixel_value, "out of range [0, 255]. Skipping.")

        print("[Mojo] Histogram calculation complete. Output list length:", len(histogram_pylist))
        return histogram_pylist.to_object()

    except e:
        print("[Mojo] Error in calculate_histogram_mojo:", e)
        # Return an empty list to indicate failure
        return Python.List[IntLiteral]().to_object()


fn calculate_bounding_box_mojo(flat_u8_list_obj: PythonObject, height: Int, width: Int) -> PythonObject:
    """
    Calculates the bounding box of non-zero pixels in a flattened grayscale image.

    Args:
        flat_u8_list_obj: A PythonObject representing a Python list of uint8 pixel values.
        height: The height of the original image.
        width: The width of the original image.

    Returns:
        A PythonObject representing a Python tuple (min_r, min_c, max_r, max_c)
        or Python.None if no non-zero pixels are found.
    """
    print("[Mojo] calculate_bounding_box_mojo called. Image dims:", height, "x", width)

    try:
        u8_input_list = Python.List[IntLiteral](flat_u8_list_obj)
        let num_pixels = len(u8_input_list)

        if num_pixels != height * width:
            print("[Mojo] BBox Error: Pixel count mismatch. Expected", height * width, ", got", num_pixels)
            return Python.None  # Return Python None object

        var min_r: Int = height
        var min_c: Int = width
        var max_r: Int = -1
        var max_c: Int = -1
        var found_pixel: Bool = False

        for i in range(num_pixels):
            let pixel_value = u8_input_list[i]
            if pixel_value != 0: # Assuming non-zero means foreground
                found_pixel = True
                let r = i // width
                let c = i % width

                if r < min_r: min_r = r
                if c < min_c: min_c = c
                if r > max_r: max_r = r
                if c > max_c: max_c = c

        if not found_pixel:
            print("[Mojo] BBox: No non-zero pixels found.")
            return Python.None

        print("[Mojo] BBox found: (", min_r, ",", min_c, ",", max_r, ",", max_c, ")")

        # Create a Python tuple to return
        # Need to convert Int to PythonObject (IntLiteral or generic Python Int)
        let py_min_r = PythonObject.from_kernel(min_r)
        let py_min_c = PythonObject.from_kernel(min_c)
        let py_max_r = PythonObject.from_kernel(max_r)
        let py_max_c = PythonObject.from_kernel(max_c)

        # Construct Python tuple
        let result_tuple = Python.Tuple(py_min_r, py_min_c, py_max_r, py_max_c)
        return result_tuple.to_object()

    except e:
        print("[Mojo] Error in calculate_bounding_box_mojo:", e)
        return Python.None
