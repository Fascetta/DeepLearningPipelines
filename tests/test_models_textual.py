from my_lib import *

def test_model_textual():
    model = NLPClassificationModel(model_name="gpt2", num_classes=4, pretrained=True)