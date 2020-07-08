import torch
import json
from transformers import DistilBertModel


class DistilBertClass(torch.nn.Module):
    def __init__(self):
        super(DistilBertClass, self).__init__()
        self.model_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dense = torch.nn.Linear(768, 768)
        self.drop_out = torch.nn.Dropout(0.3)
        self.out_proj = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.model_layer(input_ids=input_ids, attention_mask=attention_mask)
        output_2 = self.drop_out(output_1[0])
        output_3 = self.dense(output_2)
        output_4 = torch.tanh(output_3)
        output_5 = self.drop_out(output_4)
        output = self.out_proj(output_5)
        return output