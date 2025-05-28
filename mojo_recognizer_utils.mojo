from python import Python
from python import PythonObject

fn example_mojo_tensor_operation(input_data: PythonObject) -> PythonObject:
    """
    Processes a 1D array/tensor of integers (passed as a Python list).
    Performs element-wise multiplication by 2 and adds 5.
    Returns the processed data as a new Python list.
    """
    print("[Mojo] Function 'example_mojo_tensor_operation' called.")

    try:
        # Convert the PythonObject to a Python List of IntLiteral
        # Note: Mojo's Python interop for generic types like List[Int] can be specific.
        # Using IntLiteral for elements as it's often more directly translatable for simple numbers.
        py_list = Python.List[IntLiteral](input_data)
        let length = len(py_list)
        print("[Mojo] Input list length:", length)

        var new_py_list = Python.List[IntLiteral]()

        for i in range(length):
            let current_val = py_list[i] # This should be an IntLiteral
            let processed_val = current_val * 2 + 5
            new_py_list.append(processed_val)
            # print("[Mojo] Processing element", i, ": original =", current_val, ", processed =", processed_val) # Optional: verbose logging

        print("[Mojo] Finished processing. Output list length:", len(new_py_list))
        return new_py_list.to_object()

    except e:
        print("[Mojo] Error in example_mojo_tensor_operation:", e)
        # Return an empty list or raise an error that Python can catch
        # For now, returning an empty list to demonstrate error handling on Mojo side.
        var error_list = Python.List[IntLiteral]()
        return error_list.to_object()

# Main function for basic testing of the Mojo file itself (optional)
fn main():
    print("[Mojo] Main function called for testing mojo_recognizer_utils.mojo")
    # Create a dummy Python list for testing within Mojo
    var test_list = Python.List[IntLiteral]()
    test_list.append(1)
    test_list.append(2)
    test_list.append(3)
    
    print("[Mojo] Test input list created:", test_list)
    
    let result_obj = example_mojo_tensor_operation(test_list.to_object())
    
    # Try to convert result back to a list to inspect
    let result_list = Python.List[IntLiteral](result_obj)
    print("[Mojo] Test output list from function:", result_list)
    for i in range(len(result_list)):
        print("[Mojo] Result element", i, ":", result_list[i])
