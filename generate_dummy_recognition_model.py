import onnx
from onnx import helper
from onnx import TensorProto
import os

def generate_model(model_path):
    # Define input and output
    X = helper.make_tensor_value_info('input_image', TensorProto.FLOAT, [None, None]) # Batch size, num_features (flexible)
    Y = helper.make_tensor_value_info('output_text_representation', TensorProto.FLOAT, [None, None]) # Outputting floats for simplicity

    # Define a constant tensor to multiply by
    const_value = 2.0
    constant_tensor = helper.make_tensor(
        name='multiplier_const',
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[const_value]
    )

    # Create a 'Mul' node
    mul_node = helper.make_node(
        'Mul',
        inputs=['input_image', 'multiplier_const'],
        outputs=['output_text_representation'],
    )

    # Create the graph (model)
    graph_def = helper.make_graph(
        [mul_node],
        'dummy-recognition-model',
        [X], # inputs
        [Y], # outputs
        [constant_tensor] # initializer
    )

    # Create the model
    # Specify opset version (e.g., 15) and IR version for broader compatibility.
    opset_imports = [helper.make_opsetid("", 15)]
    model_def = helper.make_model(graph_def, producer_name='onnx-dummy-generator-recognition', opset_imports=opset_imports, ir_version=8)

    # Check the model
    onnx.checker.check_model(model_def)

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    onnx.save(model_def, model_path)
    print(f"Dummy recognition ONNX model saved to {model_path}")

if __name__ == "__main__":
    output_dir = "models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_filepath = os.path.join(output_dir, "dummy_recognition_model.onnx")
    generate_model(model_filepath)
