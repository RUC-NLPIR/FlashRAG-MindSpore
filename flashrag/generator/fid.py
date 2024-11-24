import types
import mindspore as ms
import mindspore.nn as nn
import mindnlp.transformers as tfs
import mindspore.ops as ops

class FiDT5(tfs.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].shape[0], -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].shape[0], -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # input ids : bs, n, seq_len -> bs, n*seq_len
    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # (bs, n, seq_len) -> (bs, n*seq_len) 
            # inputs might have already be resized in the generate method
            if input_ids.ndim() == 3:
                self.encoder.n_passages = input_ids.shape[1]
            input_ids = input_ids.view(input_ids.shape[0], -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.shape[0], -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, **kwargs):
        self.encoder.n_passages = input_ids.shape[1]
        return super().generate(
            input_ids=input_ids.view(input_ids.shape[0], -1),
            attention_mask=attention_mask.view(attention_mask.shape[0], -1),
            **kwargs,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.CellList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        ms.load_param_into_net(self,state_dict)
        # self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def tie_weights(self):
        pass  


class CheckpointWrapper(nn.Cell):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            self.module.recompute()

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
            empty = ms.tensor(
                [],
                dtype=ms.float16)
            output = tuple(x if x is not None else empty for x in output)
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.CellList(block)
    t5stack.block = block

class FiDBart(tfs.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()        
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        
        if input_ids != None:
            # (bs, n, seq_len) -> (bs, n*seq_len) 
            # inputs might have already be resized in the generate method
            if input_ids.ndim() == 3:
                self.model.encoder.n_passages = input_ids.shape[1]
                input_ids = input_ids.view(input_ids.shape[0], -1)

        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.shape[0], -1) 
        
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, attention_mask, **kwargs):
        self.model.encoder.n_passages = input_ids.shape[1]
        return super().generate(
            input_ids=input_ids.view(input_ids.shape[0], -1),
            attention_mask=attention_mask.view(attention_mask.shape[0],-1),
            **kwargs)

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.model.encoder = EncoderWrapper(self.model.encoder)
        
    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load bart weights.
        """
        self.model.encoder = self.model.encoder.encoder
        block = []
        for mod in self.model.encoder.layers:
            block.append(mod)
        block = nn.CellList(block)
        self.model.encoder.layers = block
        
    def load_pretrained_model(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict) 
        self.wrap_encoder() 
    def tie_weights(self):
        pass  

class EncoderWrapper(nn.Cell):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
        
        try:
            self.main_input_name = encoder.main_input_name
        except:
            pass
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        # outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages*passage_length, -1)
        return outputs
