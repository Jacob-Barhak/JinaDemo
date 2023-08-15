""" Jina Sample Executor
    Written by Jacob Barhak for "an evening of python coding" demo
"""

from docarray import DocumentArray, Document
from jina import Executor, Flow, requests
import torchvision
import torch
import json

class PrepImg(Executor):
    """
    Pre-process an image or video to have tensors with correct size
    """

    # noinspection PyUnusedLocal
    @requests
    async def dummy(self, docs: DocumentArray, **kwargs):
        def prep_image_for_ai(image_doc):
            """
            prepare image document for AI
            :param image_doc: the document with a tensor of an image
            :return:
            """
            # reduce dim to 200 and normalize
            image_doc.set_image_tensor_shape(shape=(200, 200))
            image_doc.set_image_tensor_normalization()
            image_doc.set_image_tensor_channel_axis(-1, 0)

        # look through files and process only images and videos
        for d in docs:
            # images get standardized and prepared for AI
            if 'image' in d.mime_type:
                d.load_uri_to_image_tensor()
                prep_image_for_ai(d)
            elif 'video' in d.mime_type:
                # video key frames get extracted and stored as images in chunks and prepared for AI
                d.load_uri_to_video_tensor(only_keyframes=True)
                video_frames = d.tensor.shape[0]
                for frame_number in range(video_frames):
                    image_tensor = d.tensor[frame_number]
                    image = Document(tensor=image_tensor, uri=f"{d.uri}-keyframe:{frame_number}")
                    prep_image_for_ai(image)
                    d.chunks.append(image)
                # release memory
                d.tensor = None


class ClassifyImg(Executor):
    """classifies images and stores the best classifications"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.number_of_classifications = 10
        self.model = torchvision.models.resnet50(pretrained=True)
        with open("imagenet_class_index.json") as classes_file:
            class_dict = json.load(classes_file)
            self.classes = [class_dict[key][1].strip() for key in class_dict]

    # noinspection PyUnusedLocal
    @requests
    async def dummy(self, docs: DocumentArray, **kwargs):
        def process_docs(image_doc):
            """
            Processes an image document and classify it, report the best classifications and confidences
            :param image_doc:
            :return:
            """
            image_doc.embed(self.model)
            # release memory
            image_doc.tensor = None
            # noinspection PyUnresolvedReferences
            probabilities = torch.nn.functional.softmax(image_doc.embedding, dim=0)
            # show top n classes per image
            top_probabilities, top_ids = torch.topk(probabilities, self.number_of_classifications)
            top_strings = [f" class{number} : {self.classes[top_id]} , confidence: {top_probability}"
                           for (number, (top_probability, top_id)) in enumerate(zip(top_probabilities, top_ids))]
            classifications = '\n'.join(top_strings)
            image_doc.text = \
                f"{image_doc.uri} - classifications:\n{classifications}"

        for doc in docs:
            if 'image' in doc.mime_type:
                process_docs(doc)
            elif 'video' in doc.mime_type:
                summary = ""
                for d1 in doc.chunks:
                    process_docs(d1)
                    summary = summary + d1.text + '\n'
                doc.text = summary


flow = (
    Flow(port_expose=12345)
    .add(uses=PrepImg)
    .add(uses=ClassifyImg, replicas=3)
)

if __name__ == '__main__':
    with flow:
        flow.block()
