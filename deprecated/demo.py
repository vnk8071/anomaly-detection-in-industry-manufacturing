import argparse
import numpy as np

import gradio as gr
from custom_inference import TorchInferencer

user_select = {
    'aqa': ['configs/patchcore_aqa.yaml', 'models/patchcore_aqa.ckpt'],
    'grid': ['configs/patchcore_grid.yaml', 'models/patchcore_grid.ckpt'],
    'hazelnut': ['configs/patchcore_hazelnut.yaml', 'models/patchcore_hazelnut.ckpt'],
    'metalnut': ['configs/patchcore_metalnut.yaml', 'models/patchcore_metalnut.ckpt']
}

def inference(type, image):
    inferencer = TorchInferencer(config=user_select[type][0], model_source=user_select[type][1])
    predictions = inferencer.predict(image=image)
    return (predictions.pred_score, predictions.pred_label, predictions.heat_map, predictions.pred_mask, predictions.segmentations)

if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=9, css="{text-align: center}"):
                title = gr.Markdown("<br><h1>Anomaly Detection on Industry Images</h1>")
            with gr.Column(min_width=50):
                fpt_logo = gr.Image('static_images/FPT-Logo.png', label='fpt')
        description = gr.Markdown("MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. \
                                    It contains over 5000 high-resolution images divided into fifteen different object and texture categories. \
                                    Each category comprises a set of defect-free training images and a test set of images with various kinds of \
                                    defects as well as images without defects. <a href='https://www.mvtec.com/company/research/datasets/mvtec-ad' style='color: #6495ED;'>Dataset</a>")
        patchcore_flow = gr.Image('static_images/patchcore-model.jpg', label='Flow')
        flow = gr.Markdown("<h5>Flow model: <a href='https://openaccess.thecvf.com//content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf' style='color: #6495ED;'>Paper Link</a></h5>")
        title_demo = gr.Markdown("<h2><center>Demo App</center></h2>")
        with gr.Row():
            with gr.Column():
                inputs=[
                    gr.Dropdown(["aqa", "grid", "hazelnut", "metalnut"], label="Choose object and texture category:", value="aqa"),
                    gr.inputs.Image(
                        shape=None, image_mode="RGB", source="upload", tool="editor", type="numpy", label="Image"
                    ),
                ]
                predict = gr.Button("Predict")  
                examples = gr.Examples(examples=[["aqa", "samples/1_0.png"], ["grid", "data/grid/test/bent/006.png"], ["hazelnut", "data/hazelnut/test/crack/004.png"], ["metalnut", "data/metal_nut/test/bent/005.png"]], inputs=inputs)
            with gr.Column():
                outputs=[
                    gr.Text(label="Anomaly Score:"),
                    gr.Text(label="Target label:"),
                    gr.outputs.Image(type="numpy", label="Predicted Heat Map"),
                    gr.outputs.Image(type="numpy", label="Predicted Mask"),
                    gr.outputs.Image(type="numpy", label="Segmentation Result"),
                ]
        
        predict.click(inference, inputs, outputs)
        footer = gr.Markdown("<h4><center>DCI-CTA project</center></h4>")
    demo.launch(debug=True)