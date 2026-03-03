import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def serialize_message(messages):
    parts = []
    for m in messages:
        role = m["role"].upper()
        content = m["content"].strip()
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


class EncoderBinaryClassifier(nn.Module):
    def __init__(self, encoder_name="roberta-base", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls).squeeze(-1)
        return logits


def aggregate_chunk_logits(chunk_logits, sample_map, batch_size):
    out = torch.full((batch_size,), -1e9, device=chunk_logits.device)
    out.scatter_reduce_(0, sample_map, chunk_logits, reduce="amax", include_self=True)
    return out


class SafetyScorer:
    def __init__(
        self,
        checkpoint_path,
        encoder_name="roberta-base",
        max_length=512,
        stride=128,
        dropout=0.1,
        device=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = EncoderBinaryClassifier(encoder_name, dropout=dropout)

        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.stride = stride

    @torch.no_grad()
    def predict_proba(self, examples, batch_size=16):
        texts = [serialize_message(msgs) for msgs in examples]

        probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )

            sample_map = enc.pop("overflow_to_sample_mapping").long().to(self.device)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            chunk_logits = self.model(**enc)
            B = len(batch_texts)
            ex_logits = aggregate_chunk_logits(chunk_logits, sample_map, B)

            probs.append(torch.sigmoid(ex_logits).cpu())

        return torch.cat(probs, dim=0)