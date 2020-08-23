import torch.nn
from transformers import RobertaModel


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.model_layer = RobertaModel.from_pretrained("roberta-base")
        self.dense = torch.nn.Linear(768, 768)
        self.drop_out = torch.nn.Dropout(0.3)
        self.out_proj = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.model_layer(input_ids=input_ids, attention_mask=attention_mask)
        output_2 = self.drop_out(output_1[1])
        output_3 = self.dense(output_2)
        output_4 = torch.tanh(output_3)
        output_5 = self.drop_out(output_4)
        output = self.out_proj(output_5)
        return output
