import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

MODEL_NM = "microsoft/phi-2"
basemodel = AutoModelForCausalLM.from_pretrained(MODEL_NM, torch_dtype=torch.float32, device_map="cuda", trust_remote_code=True)
print(basemodel)

class PhiForSequenceClassificationModified(PhiPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = NUM_LABELS#changed
        self.model = basemodel.transformer#changed
        self.score = nn.Linear(basemodel.config.hidden_size, NUM_LABELS, bias=False)#changed
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embd.wte#changed

    def set_input_embeddings(self, value):
        self.model.embd.wte = value#changed

    @add_start_docstrings_to_model_forward("PHI_INPUTS_DOCSTRING")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            
        )
        hidden_states = model_outputs#changed
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        
        if input_ids is not None:
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                logits.device
            )
        else:
            sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (pooled_logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )#changed    