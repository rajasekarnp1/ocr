import onnx
from onnx import helper
from onnx import TensorProto
import os

def generate_model(model_path):
    # Define input and output
    X = helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [None, None]) # Batch size, num_features (flexible)
    Y = helper.make_tensor_value_info('output_tensor', TensorProto.FLOAT, [None, None])

    # Define a constant tensor to add
    const_value = 1.0
    constant_tensor = helper.make_tensor(
        name='const_tensor',
        data_type=TensorProto.FLOAT,
        dims=[1], # Scalar, but ONNX expects it as a 1D tensor for broadcasting
        vals=[const_value]
    )

    # Create an 'Add' node
    add_node = helper.make_node(
        'Add',
        inputs=['input_tensor', 'const_tensor'],
        outputs=['output_tensor'],
    )

    # Create the graph (model)
    graph_def = helper.make_graph(
        [add_node],
        'dummy-geometric-model',
        [X], # inputs
        [Y], # outputs
        [constant_tensor] # initializer (weights/constants)
    )

    # Create the model
    # Specify opset version (e.g., 15) and IR version for broader compatibility.
    opset_imports = [helper.make_opsetid("", 15)]
    model_def = helper.make_model(graph_def, producer_name='onnx-dummy-generator-geometric', opset_imports=opset_imports, ir_version=8)

    # Check the model
    onnx.checker.check_model(model_def)

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    onnx.save(model_def, model_path)
    print(f"Dummy geometric ONNX model saved to {model_path}")

if __name__ == "__main__":
    output_dir = "models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_filepath = os.path.join(output_dir, "dummy_geometric_model.onnx")
    generate_model(model_filepath)
