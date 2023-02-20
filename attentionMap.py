from transformers import EsmTokenizer, EsmModel, EsmConfig
import torch
import matplotlib.pyplot as plt
import numpy

model_name = "facebook/esm2_t33_650M_UR50D"
# model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)
model.config.output_attentions = True
print(model.config)

data = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", 
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE", 
        "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "K A <mask> I S Q"]

inputs = tokenizer(data, return_tensors="pt", padding = True, add_special_tokens = True)
b, t = inputs['input_ids'].size() 
batch_lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(1)  # //tensor([67, 73, 73,  8])

outputs = model(**inputs, output_attentions = True)
'''
====================================================================================================
Representations
====================================================================================================
'''
token_representations = outputs.last_hidden_state
b, t, k = token_representations.size()

sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

sequence_representations = torch.cat([item[None] for item in sequence_representations], dim=0)
assert b, k == sequence_representations.size()

'''
====================================================================================================
Attention Maps
====================================================================================================
'''
NUM_LAYER = model.config.num_hidden_layers #l = 33
NUM_HEAD = model.config.num_attention_heads #h = 20

all_attentions = outputs.attentions  
#[(batch_size, num_heads, seq_length, seq_length),...()] len = num_hidden_layer
l = len(all_attentions)
b, h, t, t = all_attentions[0].size()

all_attentions = torch.cat([item for item in all_attentions], axis = 1)  # (b, l*h, t, t)

attention_map = {}
for label, seq_len in enumerate(batch_lens): 
         attention_map[f"protein_{label}"] = all_attentions[label, :, 1: seq_len - 1, 1: seq_len - 1]

print([attention_map[f"protein_{i}"].size() for i in range(len(batch_lens))])   
# //[torch.Size([660, 65, 65]), torch.Size([660, 71, 71]), torch.Size([660, 71, 71]), torch.Size([660, 6, 6])] 

'''
====================================================================================================
Attention Map Visualization
====================================================================================================
'''
sample_map = attention_map["protein_1"][0, :, :]

plt.matshow(sample_map.detach().numpy())
plt.title("sample attention map of protein_1")


