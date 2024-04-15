import dash
from dash import dcc, html, Input, Output, State
import requests
import base64
from PIL import Image
from io import BytesIO
import time
from dash.dcc import Interval
from typing import Tuple, List

## Defining app layout
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        html.H1("Cardiac Device Detection and Classification"),  # Title
        dcc.Upload(
            id="upload-image", children=html.Button("Upload Image"), multiple=False
        ),
        html.Div(id="output-container"),  # Upload button
        html.Div(id="elapsed-time-container"),  # Placeholder Div for elapsed time
        Interval(
            id="interval-component", interval=1 * 100, n_intervals=0  # in milliseconds
        ),  # Add Interval component to update processing time
        dcc.Store(
            id="image-buffer", storage_type="memory"
        ),  # Memory buffer to separate upload and backend response event triggets
    ]
)


@app.callback(
    [
        Output("output-container", "children", allow_duplicate=True),
        Output("image-buffer", "data"),
    ],
    [Input("upload-image", "contents")],
    [State("upload-image", "filename"), State("upload-image", "last_modified")],
    prevent_initial_call=True,
)
def receive_input(
    contents: str, filename: str, last_modified: int
) -> Tuple[List[html.Div], str]:
    """Receives the input image content, filename, and last modified timestamp.
    
    Decodes the image content, displays the input image, and returns the image content.

    Args:
        contents (str): The base64-encoded string representing the image content.
        filename (str): The name of the uploaded file.
        last_modified (int): The last modified timestamp of the uploaded file.

    Returns:
        Tuple[List[html.Div], str]: A tuple containing a list of HTML div elements
        representing the output container and the image content.
    """
    output_content = []
    image_string = None
    if contents is not None:
        content_type, image_string = contents.split(",")
        decoded = base64.b64decode(image_string)

        input_image = Image.open(BytesIO(decoded))
        input_image_width = min(
            input_image.width, int(0.4 * 1000)
        )  # Limit input image width for lower screen resolutions
        input_image_height = input_image.height * (
            input_image_width / input_image.width
        )
        input_image_base64 = base64.b64encode(decoded).decode()
        input_image_html = html.Div(
            [
                html.H2("Input Image"),  # Header for input image
                html.Img(
                    src="data:image/png;base64,{}".format(input_image_base64),
                    width=input_image_width,
                ),
            ],
            style={"display": "inline-block", "margin-right": "20px"},
        )
        output_content = [input_image_html]
    return output_content, image_string


@app.callback(
    [Output("output-container", "children", allow_duplicate=True)],
    [Input("image-buffer", "data")],
    prevent_initial_call=True,
)
def process_upload_output(data: str) -> List[html.Div]:
    """Processes the uploaded image content and displays the output image.

    Decodes the image content, sends it to the backend for processing, and displays
    the input and output images along with the prediction text.

    Args:
        data (str): The base64-encoded string representing the image content.

    Returns:
        List[html.Div]: A list of HTML div elements representing the output container.
    """
    output_content = []
    if data is not None:
        decoded = base64.b64decode(data)
        # Display input image
        input_image = Image.open(BytesIO(decoded))
        input_image_width = min(
            input_image.width, int(0.4 * 1000)
        )  # Limit input image width for lower screen resolutions
        input_image_height = input_image.height * (
            input_image_width / input_image.width
        )
        input_image_base64 = base64.b64encode(decoded).decode()
        input_image_html = html.Div(
            [
                html.H2("Input Image"),
                html.Img(
                    src="data:image/png;base64,{}".format(input_image_base64),
                    width=input_image_width,
                ),
            ],
            style={"display": "inline-block", "margin-right": "20px"},
        )

        # Reset elapsed time
        app.start_time = time.time()

        # Send image to the backend for processing
        response = requests.post(
            "http://backend:8000/upload/", files={"file": ("image.png", decoded)}
        )

        if response.status_code == 200:
            data = response.json()

            # Display output image
            output_image = Image.open(BytesIO(base64.b64decode(data["image"])))
            output_image_height = input_image_height  # Mathing output image height with input height for good rendering
            output_image_width = output_image.width * (
                output_image_height / output_image.height
            )

            output_image_base64 = data["image"]
            output_image_html = html.Div(
                [
                    html.H2(f"Detected Device"),  # Header for output image
                    html.Img(
                        src="data:image/png;base64,{}".format(output_image_base64),
                        width=output_image_width,
                    ),
                ],
                style={"display": "inline-block", "margin-left": "20px"},
            )

            # Display text generated by backend
            text_html = html.Div([html.P(f"Prediction: {data['text']}")])
            # Update the output-content-container with the output image and text
            output_content = [input_image_html, output_image_html, text_html]
            del app.start_time
    return [output_content]


@app.callback(
    Output("elapsed-time-container", "children"),
    [Input("interval-component", "n_intervals")],
)
def update_processing_time(n: int) -> List[html.P]:
    """Updates the processing time displayed on the frontend.

    Calculates the elapsed time since the start of processing and returns
    a list of HTML p elements containing the processing time.

    Args:
        n (int): The number of intervals.

    Returns:
        List[html.P]: A list of HTML p elements representing the elapsed time.
    """
    if hasattr(app, "start_time"):
        # Calculate elapsed time
        elapsed_time = time.time() - app.start_time
        # Update processing time every second
        return [html.P(f"Processing... Time elapsed: {elapsed_time:.1f} seconds.")]
    else:
        return []


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
